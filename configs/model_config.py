"""
Model configuration for fashion trend prediction.
"""

def get_default_config():
    config = {
        # Input configuration
        'input_size': (1280, 1280),
        'input_channels': 3,
        
        # Backbone configuration
        'backbone': {
            'type': 'spinenet',
            'min_level': 3,
            'max_level': 7,
            'init_channels': 64,
        },
        
        # FPN configuration
        'fpn': {
            'in_channels_list': [256, 256, 256, 256, 256],  # One for each level
            'out_channels': 256,
        },
        
        # RPN configuration
        'rpn': {
            'anchor_sizes': ((32,), (64,), (128,), (256,), (512,)),
            'aspect_ratios': ((0.5, 1.0, 2.0),) * 5,
            'rpn_pre_nms_top_n_train': 2000,
            'rpn_post_nms_top_n_train': 1000,
            'rpn_pre_nms_top_n_test': 1000,
            'rpn_post_nms_top_n_test': 500,
        },
        
        # RoI configuration
        'roi': {
            'box_roi_pool': {
                'output_size': 7,
                'sampling_ratio': 2,
            },
            'box_head': {
                'fc_features': 1024,
            },
        },
        
        # Attribute head configuration
        'attribute_head': {
            'in_channels': 256,
            'num_attributes': 294,  # iMaterialist attributes
            'hidden_dim': 1024,
        },
        
        # Training configuration
        'training': {
            'batch_size': 8,
            'epochs': 50,
            'learning_rate': 0.02,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'lr_scheduler': {
                'milestones': [30, 40],
                'gamma': 0.1,
            },
            'focal_loss': {
                'alpha': 0.25,
                'gamma': 2.0,
            },
        },
        
        # Augmentation configuration
        'augmentation': {
            'min_scale': 0.5,
            'max_scale': 2.0,
            'autoaugment_policy': 'v3',
        },
    }
    
    return config 