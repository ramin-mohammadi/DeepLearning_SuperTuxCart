from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html#torch.nn.functional.nll_loss
        # look at example in link
        
        """
        Loss function expects:
        - a 2D tensor of shape (batch_size, num_classes) as input for the predictions (logits) and a 
        1D tensor of shape (batch_size,) for the targets (integer class labels).
        -> But currently logits is shape (batch, classes, width, height) bc last layer is convolution
        """
        return nn.functional.nll_loss(nn.functional.log_softmax(logits, dim=1), target)
        #return nn.functional.cross_entropy(logits, target)
        
class DetectionLoss(nn.Module):
    def forward(self, logits: torch.Tensor, raw_depth: torch.Tensor, target_depth: torch.LongTensor, target_track: torch.LongTensor) -> torch.Tensor:
        #x = torch.max(torch.nn.functional.softmax(logits, dim=1), dim=1).values
        return nn.functional.cross_entropy(logits, target_track.float()) + nn.functional.mse_loss(raw_depth, target_depth)
        #return nn.functional.cross_entropy(logits, target_track) + nn.functional.mse_loss(raw_depth, target_depth)


class Classifier(nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            
            kernel_size = 3
            padding = (kernel_size-1)//2
          
            self.model = torch.nn.Sequential( 
                torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                #torch.nn.LayerNorm(out_channels),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
            )

            # RESIDUAL PART solving for shape mismatch
            if in_channels != out_channels: # add linear layer to get shapes to match
                #self.skip = torch.nn.Linear(in_channels, out_channels)
                self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            else: # otherwise no reshape needed so identity is like x * 1
                self.skip = torch.nn.Identity()

        def forward(self, x):
          return self.skip(x) + self.model(x) # RESIDUAL PART is self.skip(x) + 
           

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        channels_l0 = 64,
        n_blocks=2,
    ):
        """
        A convolutional network for image classification.

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        
        cnn_layers = [
          # first input is an image (3 channels)
          torch.nn.Conv2d(3, channels_l0, kernel_size=11, stride=2, padding=5),
          torch.nn.ReLU(),
        ]
        
        c1 = channels_l0
        
        for _ in range(n_blocks):
          c2=c1 * 2
          cnn_layers.append(self.Block(c1, c2, stride=2))
          c1=c2 # input channel of next layer must match output of previous layer
        
        
        """
        classifier
        - FIRST OPTION: you can either take outputs of conv net and average pool them all together and then apply classifier
        - SECOND OPTION: Or you can add a 1x1 convolution (essentially matrix mult applied element wise to all outputs) and then you would average
        - first option is computationally cheaper but second option nicer bc you can infer which spatial locations correspond to certain classifications
        """
        
        """
        cnn_layers.append(torch.nn.Flatten()) # if doing flatten, will not work for conv2d following it bc it expects 4d input
        cnn_layers.append(torch.nn.Linear(16384, num_classes, bias=False)) 
        """
        
        
        # second option
        cnn_layers.append(torch.nn.Conv2d(c1, num_classes, kernel_size=1)) # get channels to be num_classes
        
        
        # if you want to average the outputs of the classifier inside of your network:
        cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1))
        self.network = torch.nn.Sequential(*cnn_layers)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, h, w) image

        Returns:
            tensor (b, num_classes) logits
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # TODO: replace with actual forward pass
        #logits = torch.randn(x.size(0), 6)

        #return logits
        
        # Flatten (batch, num_classes, 1, 1) to (batch_size, num_classes) 
        
        return self.network(z).squeeze(-1).squeeze(-1)
        

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Used for inference, returns class labels
        This is what the AccuracyMetric uses as input (this is what the grader will use!).
        You should not have to modify this function.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
        """
        return self(x).argmax(dim=1)







class Detector(torch.nn.Module):
    class Down_Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            
            kernel_size = 3
            padding = (kernel_size-1)//2
          
            self.model = torch.nn.Sequential( 
                torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                #torch.nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding),
                #torch.nn.LayerNorm(out_channels),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
                torch.nn.ReLU(),
                #torch.nn.MaxPool2d(kernel_size=2, stride=2)
            )

            # RESIDUAL PART solving for shape mismatch
            if in_channels != out_channels: # add linear layer to get shapes to match
                #self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride*2)
                self.skip = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            else: # otherwise no reshape needed so identity is like x * 1
                self.skip = torch.nn.Identity()

        def forward(self, x):
          return self.skip(x) + self.model(x) # RESIDUAL PART is self.skip(x) + 
          #return self.model(x)
          
    class Up_Block(torch.nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            
            kernel_size = 3
            padding = (kernel_size-1)//2
          
            self.model = torch.nn.Sequential( 
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1),
                torch.nn.BatchNorm2d(out_channels),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
                torch.nn.ReLU(),
                #torch.nn.ConvTranspose2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=1),
                #torch.nn.ReLU(),
                torch.nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
                torch.nn.ReLU(),
            )

            # RESIDUAL PART solving for shape mismatch
            if in_channels != out_channels: # add linear layer to get shapes to match
                #self.skip = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride*2, output_padding=3)
                self.skip = torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, output_padding=1)

            else: # otherwise no reshape needed so identity is like x * 1
                self.skip = torch.nn.Identity()

        def forward(self, x):
          return self.skip(x) + self.model(x) # RESIDUAL PART is self.skip(x) + 
          


    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        channels_l0 = 16,
        n_blocks=3,
    ):
        """
        A single model that performs segmentation and depth regression

        Args:
            in_channels: int, number of input channels
            num_classes: int
        """
        super().__init__()

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

        # TODO: implement
        #pass
        
        cnn_layers = [
          # first input is an image (3 channels)
          torch.nn.Conv2d(3, channels_l0, kernel_size=11, stride=2, padding=5),
          torch.nn.ReLU(),
        ]
        
        c1 = channels_l0
        
        # down conv blocks
        for _ in range(n_blocks):
          c2=c1 * 2
          cnn_layers.append(self.Down_Block(c1, c2, stride=2))
          c1=c2 # input channel of next layer must match output of previous layer
          
        
        #cnn_layers.append(torch.nn.Conv2d(c1, c1, 3, 1, 1))
        #cnn_layers.append(torch.nn.Conv2d(c1, c1, 3, 1, 1))

          
        # up conv blocks
        for _ in range(n_blocks):
          c2=c1 // 2
          #c2 = channels_l0
          cnn_layers.append(self.Up_Block(c1, c2, stride=2))
          c1=c2 # input channel of next layer must match output of previous layer
        
        
        """
        classifier
        - FIRST OPTION: you can either take outputs of conv net and average pool them all together and then apply classifier
        - SECOND OPTION: Or you can add a 1x1 convolution (essentially matrix mult applied element wise to all outputs) and then you would average
        - first option is computationally cheaper but second option nicer bc you can infer which spatial locations correspond to certain classifications
        """
        
        self.num_classes = num_classes
        self.c1 = c1
        
        self.conv1 = torch.nn.ConvTranspose2d(self.c1, self.num_classes, kernel_size=1, stride=2, output_padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(self.c1, 1, kernel_size=1, stride=2, output_padding=1)
        
        """
        # second option
        cnn_layers.append(torch.nn.Conv2d(c1, num_classes, kernel_size=1)) # get channels to be num_classes
        
        
        # if you want to average the outputs of the classifier inside of your network:
        cnn_layers.append(torch.nn.AdaptiveAvgPool2d(1)) # parameter is # channels we expect as input
        """
        
        self.network = torch.nn.Sequential(*cnn_layers)
        
        
        self.track_head = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.c1, self.num_classes, kernel_size=1, stride=2, output_padding=1),
            #torch.nn.Conv2d(self.num_classes, self.num_classes, 1, 1),
        )
        
        self.depth_head = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.c1, 1, kernel_size=1, stride=2, output_padding=1),
            #torch.nn.Conv2d(1, 1, 1, 1),
        )
        
        

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used in training, takes an image and returns raw logits and raw depth.
        This is what the loss functions use as input.

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.FloatTensor, torch.FloatTensor):
                - logits (b, num_classes, h, w)
                - depth (b, h, w)
        """
        # optional: normalizes the input
        z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        """
        # TODO: replace with actual forward pass
        logits = torch.randn(x.size(0), 3, x.size(2), x.size(3))
        raw_depth = torch.rand(x.size(0), x.size(2), x.size(3))
        """
            
        net = self.network(z)
        
        
        #logits = conv1(net)
        #raw_depth = conv2(net)

        # turn logits from shape (batch, 3, 96, 128) to (batch, 96, 128) by taking max prob in dim 1
        # and turn raw_depth from shape (batch , 1, 96, 128) to (batch, 96, 128)
        #return torch.max(torch.nn.functional.softmax(self.conv1(net), dim=1), dim=1).values, self.conv2(net).squeeze(1)
        #return self.conv1(net).argmax(dim=1).float(), self.conv2(net).squeeze(1)
        return self.track_head(net).argmax(dim=1).float(), self.depth_head(net).squeeze(1)


        
        
        #return torch.max(torch.nn.functional.softmax(self.track_head(net), dim=1), dim=1).values, self.depth_head(net).squeeze(1)
      
        
        #return self.conv1(net), self.conv2(net) # (batch, 3, 96, 128) and (batch , 1, 96, 128)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Used for inference, takes an image and returns class labels and normalized depth.
        This is what the metrics use as input (this is what the grader will use!).

        Args:
            x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            tuple of (torch.LongTensor, torch.FloatTensor):
                - pred: class labels {0, 1, 2} with shape (b, h, w)
                - depth: normalized depth [0, 1] with shape (b, h, w)
        """
        
        
        logits, raw_depth = self(x) # calls .forward()
        
        #net = self.network(x)
        
        #pred = logits.argmax(dim=1)

        # Optional additional post-processing for depth only if needed
        #depth = raw_depth

        # turn logits from shape (batch, 3, 96, 128) to (batch, 96, 128) by taking max prob in dim 1
        # and turn raw_depth from shape (batch , 1, 96, 128) to (batch, 96, 128)
        #return torch.max(torch.nn.functional.softmax(self.conv1(net), dim=1), dim=1).values, self.conv2(net).squeeze(1)
        return logits, raw_depth


MODEL_FACTORY = {
    "classifier": Classifier,
    "detector": Detector,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def debug_model(batch_size: int = 1):
    """
    Test your model implementation

    Feel free to add additional checks to this function -
    this function is NOT used for grading
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_batch = torch.rand(batch_size, 3, 64, 64).to(device)

    print(f"Input shape: {sample_batch.shape}")

    model = load_model("classifier", in_channels=3, num_classes=6).to(device)
    output = model(sample_batch)

    # should output logits (b, num_classes)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    debug_model()
