default_options = {
    # dataset config
    'batch_size':{
        'type': int,
        'default': 1024*8
    },
    'dataset': {
        'type': str,
        'nargs': 3,
        'default': ['synthetic','/localhome/zla247/theia2_data/fipt_indoor_synthetic/kitchen','/localhome/zla247/theia2_data/output/BRDF-Fipt/fipt_indoor_synthetic/kitchen']
    },
    'voxel_path': {
        'type': str,
        'default': '/localhome/zla247/theia2_data/output/BRDF-Fipt/fipt_indoor_synthetic/kitchen/vslf.npz'
    },
    'num_workers': {
        'type': int,
        'default': 12
    },
    
    # whether has part segmentation
    'has_part': {
        'type': int,
        'default': 1
    },
    

    # optimizer config
    'optimizer': {
        'type': str,
        'choices': ['SGD', 'Ranger', 'Adam'],
        'default': 'Adam'
    },
    'learning_rate': {
        'type': float,
        'default': 1e-3
    },
    'weight_decay': {
        'type': float,
        'default': 0
    },

    'scheduler_rate':{
        'type': float,
        'default': 0.5
    },
    'milestones':{
        'type': int,
        'nargs': '*',
        'default': [1000]
    },
    
    
    # reuglarization config
    'le': {
        'type': float,
        'default': 1.0
    },
    'ld': {
        'type': float,
        'default': 5e-4
    },
    'lp': {
        'type': float,
        'default': 5e-3
    },
    'ls': {
        'type': float,
        'default': 1e-3
    },
    'sigma_albedo': {
        'type': float,
        'default': 0.05/3.0
    },
    'sigma_pos': {
        'type': float,
        'default': 0.3/3.0
    }
}
