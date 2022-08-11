def get_config(args):
    if args.model == 'resnet50':
      config = {
    "model": 'conv',
    "embed_dim": 1024,
    "image_resolution": 224,
    "vision_layers": [
        3,
        4,
        6,
        3
    ],
    "vision_width": 64,
    "vision_patch_size": None,
    "context_length": 77,
    "vocab_size": 49408,
    "transformer_width": 512,
    "transformer_heads": 8,
    "transformer_layers": 12
}
        
    elif args.model == 'resnet101':
       config = {
    "model": 'conv',
    "embed_dim": 512,
    "image_resolution": 224,
    "vision_layers": [
        3,
        4,
        23,
        3
    ],
    "vision_width": 64,
    "vision_patch_size": None,
    "context_length": 77,
    "vocab_size": 49408,
    "transformer_width": 512,
    "transformer_heads": 8,
    "transformer_layers": 12
}
    
    elif args.model == 'resnet50x4':
       config = {
    "model": 'conv',
    "embed_dim": 640,
    "image_resolution": 288,
    "vision_layers": [
        4,
        6,
        10,
        6
    ],
    "vision_width": 80,
    "vision_patch_size": None,
    "context_length": 77,
    "vocab_size": 49408,
    "transformer_width": 640,
    "transformer_heads": 10,
    "transformer_layers": 12
}
        
    elif args.model == 'vit16-b':
        config = {
    "model": 'vit',
    "embed_dim": 512,
    "image_resolution": 224,
    "vision_layers": 12,
    "vision_width": 768,
    "vision_patch_size": 16,
    "context_length": 77,
    "vocab_size": 49408,
    "transformer_width": 512,
    "transformer_heads": 8,
    "transformer_layers": 12
}
    elif args.model == 'resnet50x16':
        config = {
    "model": 'conv',
    "embed_dim": 768,
    "image_resolution": 384,
    "vision_layers": [
        6,
        8,
        18,
        8
    ],
    "vision_width": 96,
    "vision_patch_size": None,
    "context_length": 77,
    "vocab_size": 49408,
    "transformer_width": 768,
    "transformer_heads": 12,
    "transformer_layers": 12
}
        
    elif args.model == 'vit32-b':
        config = {
    "model": 'vit',
    "embed_dim": 512,
    "image_resolution": 224,
    "vision_layers": 12,
    "vision_width": 768,
    "vision_patch_size": 32,
    "context_length": 77,
    "vocab_size": 49408,
    "transformer_width": 512,
    "transformer_heads": 8,
    "transformer_layers": 12
}
    return config