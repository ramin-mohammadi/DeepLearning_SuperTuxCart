import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import DetectionLoss, load_model, save_model
from .datasets.road_dataset import load_data
from .metrics import DetectionMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",  # in models.py look at model_factory
    num_epoch: int = 15,
    lr: float = 1e-3, # orig 1e-2 but abs_depth_error and tp_depth_error blew up so changed to 1e-3 and fixed issue
    batch_size: int = 32, # 128 -> had to decrease batch size bc not enough GPU memory
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs) # choose string name from model_factory in model.py of model to instantiate
    model = model.to(device)
    model.train()
    #print(model)
    
    """
    Implement the `Detector` model in `models.py`.
    Your `forward` function receives a `(B, 3, 96, 128)` image tensor as an input and should return both:
    - `(B, 3, 96, 128)` logits for the 3 classes
    - `(B, 1, 96, 128)` tensor of depths.
    
    This dataset yields a dictionary with keys `image`, `depth`, and `track` (segmentation labels).
    
    THEREFORE, an image from dataset is fed into our model and generates a label prediction and a depth prediction. 
    Then the entries of depth and track (one of 3 classes) from the dataset are the true labels.
    """

    train_data = load_data("road_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("road_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = DetectionLoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_acc": [], "train_iou": [], "train_abs_depth_error": [], "train_tp_depth_error": [],
        "val_acc": [], "val_iou": [], "val_abs_depth_error": [], "val_tp_depth_error": [],}
    detection_metric = DetectionMetric()

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()
        detection_metric.reset()

        model.train() # train mode

        for batch in train_data: # each iteration of data loader object is a batch
            """
            print(batch['image'].shape) # torch.Size([128, 3, 96, 128])
            print(batch['depth'].shape) # torch.Size([128, 96, 128])
            print(batch['track'].shape, "\n") # torch.Size([128, 96, 128])
            """
            
            image, depth, track = batch['image'].to(device), batch['depth'].to(device), batch['track'].to(device)
            
            logits, raw_depth = model(image) # bc in train() mode, the forward() function is called
            # Observe outputted values and true labels/values
            """
            print(logits.shape, raw_depth.shape, "\n") # torch.Size([128, 3, 96, 128]) torch.Size([128, 1, 96, 128]) 
            print(logits)
            print(track)
            print(raw_depth)
            print(depth)
            """
            
            
            """
            NOTE: the model now predicts a class for **every pixel** in the image. 
            -> which is why the true pred label 'track' is 96x128 
            """
            
            # Compute Loss
            loss_val = loss_func(logits, raw_depth, depth, track)

            # Compute Gradient and update weights
            optimizer.zero_grad() # MUST: reset gradients to zero or will sum up consecutive .backward()s
            loss_val.backward()
            optimizer.step()

            global_step += 1
            
            #metrics['train_acc'].append(compute_accuracy(out, label))
            detection_metric.add(preds=logits, depth_preds=raw_depth, labels=track, depth_labels=depth)
            metric_train_acc = detection_metric.compute()
            #print("train_acc: ", metric_train_acc)
            metrics['train_acc'].append(metric_train_acc['accuracy'])
            metrics['train_iou'].append(metric_train_acc['iou'])
            metrics['train_abs_depth_error'].append(metric_train_acc['abs_depth_error'])
            metrics['train_tp_depth_error'].append(metric_train_acc['tp_depth_error'])

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for batch in val_data:
                image, depth, track = batch['image'].to(device), batch['depth'].to(device), batch['track'].to(device)

                logits, raw_depth = model(image) # bc in evaluation mode, model's predict() method is called
                
                #store val acc in metrics["val_acc"]
                #metrics['val_acc'].append(compute_accuracy(out, label))
                detection_metric.add(preds=logits, depth_preds=raw_depth, labels=track, depth_labels=depth)
                metric_val_acc = detection_metric.compute()
                #print("val_acc: ", metric_val_acc)
                metrics['val_acc'].append(metric_val_acc['accuracy'])
                metrics['val_iou'].append(metric_train_acc['iou'])
                metrics['val_abs_depth_error'].append(metric_train_acc['abs_depth_error'])
                metrics['val_tp_depth_error'].append(metric_train_acc['tp_depth_error'])
                

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_train_iou = torch.as_tensor(metrics["train_iou"]).mean()
        epoch_train_abs_depth_error = torch.as_tensor(metrics["train_abs_depth_error"]).mean()
        epoch_train_tp_depth_error = torch.as_tensor(metrics["train_tp_depth_error"]).mean()
        
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        epoch_val_iou = torch.as_tensor(metrics["val_iou"]).mean()
        epoch_val_abs_depth_error = torch.as_tensor(metrics["val_abs_depth_error"]).mean()
        epoch_val_tp_depth_error = torch.as_tensor(metrics["val_tp_depth_error"]).mean()
        
        logger.add_scalar('train_accuracy', epoch_train_acc, global_step)
        logger.add_scalar('train_iou', epoch_train_iou, global_step)
        logger.add_scalar('train_abs_depth_error', epoch_train_abs_depth_error, global_step)
        logger.add_scalar('train_tp_depth_error', epoch_train_tp_depth_error, global_step)
        
        logger.add_scalar('val_accuracy', epoch_val_acc, global_step)
        logger.add_scalar('val_iou', epoch_val_iou, global_step)
        logger.add_scalar('val_abs_depth_error', epoch_val_abs_depth_error, global_step)
        logger.add_scalar('val_tp_depth_error', epoch_val_tp_depth_error, global_step)

        #raise NotImplementedError("Logging not implemented")

        # print on first, last, every 2nd epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 2 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"train_iou={epoch_train_iou:.4f} "
                f"train_abs_depth_error={epoch_train_abs_depth_error:.4f} "
                f"train_tp_depth_error={epoch_train_tp_depth_error:.4f} "
                f"val_acc={epoch_val_acc:.4f} "
                f"val_iou={epoch_val_iou:.4f} "
                f"val_abs_depth_error={epoch_val_abs_depth_error:.4f} "
                f"val_tp_depth_error={epoch_val_tp_depth_error:.4f} "
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
