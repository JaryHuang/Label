from .Resnet import ResNet18, ResNet50
from .BasicModule import BasicModule



def Switch_Model(model_name,num_classes):
    if model_name == "ResNet18":
        model =ResNet18(num_classes = num_classes, )
    
    elif model_name == "ResNet50":
        model = ResNet50(num_classes = num_classes)
    
    elif model_name == "ResNext":
        model = CifarResNeXt(cardinality=8, depth=29, nlabels=num_classes, base_width=64, widen_factor=4)

    else:
        raise Exception("Sorry, the model {} is not support in this version".format(model_name))

    return model