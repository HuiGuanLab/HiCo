root_path = "/data/xxx/datasets/skeleton/data"

class  opts_ntu_60_cross_view():

  def __init__(self):

    # Sequence based model
    self.encoder_args = {
    "t_input_size":150,
    "s_input_size":192,
    "kernel_size":5, 
    "stride":1, 
    "padding":2, 
    "factor":2,
    "hidden_size":512,
    "num_head":4,
    "num_layer":1,
    "granularity":4,
    "encoder":"Transformer",
    "num_class":60
    }
  
    # feeder
    self.train_feeder_args = {
      "data_path": root_path + "/NTU-RGB-D-60-AGCN/xview/train_data_joint.npy",
      "label_path": root_path + "/NTU-RGB-D-60-AGCN/xview/train_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-60-AGCN/xview/train_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }
   
    self.test_feeder_args = {

      'data_path': root_path + "/NTU-RGB-D-60-AGCN/xview/val_data_joint.npy",
      'label_path': root_path + "/NTU-RGB-D-60-AGCN/xview/val_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-60-AGCN/xview/val_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }

class  opts_ntu_60_cross_subject():

  def __init__(self):

    # Sequence based model
    self.encoder_args = {
    "t_input_size":150,
    "s_input_size":192,
    "kernel_size":5, 
    "stride":1, 
    "padding":2, 
    "factor":2,
    "hidden_size":512,
    "num_head":4,
    "num_layer":1,
    "granularity":4,
    "encoder":"Transformer",
    "num_class":60
    }

    # feeder
    self.train_feeder_args = {
      "data_path": root_path + "/NTU-RGB-D-60-AGCN/xsub/train_data_joint.npy",
      "label_path": root_path + "/NTU-RGB-D-60-AGCN/xsub/train_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-60-AGCN/xsub/train_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }
   
    self.test_feeder_args = {

      'data_path': root_path + "/NTU-RGB-D-60-AGCN/xsub/val_data_joint.npy",
      'label_path': root_path + "/NTU-RGB-D-60-AGCN/xsub/val_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-60-AGCN/xsub/val_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }



class  opts_ntu_120_cross_subject():
  def __init__(self):

    # Sequence based model
    self.encoder_args = {
    "t_input_size":150,
    "s_input_size":192,
    "kernel_size":5, 
    "stride":1, 
    "padding":2, 
    "factor":2,
    "hidden_size":512,
    "num_head":4,
    "num_layer":1,
    "granularity":4,
    "encoder":"Transformer",
    "num_class":120
    }

    # feeder
    self.train_feeder_args = {
      "data_path": root_path + "/NTU-RGB-D-120-AGCN/xsub/train_data_joint.npy",
      "label_path": root_path + "/NTU-RGB-D-120-AGCN/xsub/train_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-120-AGCN/xsub/train_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }
   
    self.test_feeder_args = {

      'data_path': root_path + "/NTU-RGB-D-120-AGCN/xsub/val_data_joint.npy",
      'label_path': root_path + "/NTU-RGB-D-120-AGCN/xsub/val_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-120-AGCN/xsub/val_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }

class  opts_ntu_120_cross_setup():

  def __init__(self):

    # Sequence based model
    self.encoder_args = {
    "t_input_size":150,
    "s_input_size":192,
    "kernel_size":5, 
    "stride":1, 
    "padding":2, 
    "factor":2,
    "hidden_size":512,
    "num_head":4,
    "num_layer":1,
    "granularity":4,
    "encoder":"Transformer",
    "num_class":120
    }

    # feeder
    self.train_feeder_args = {
      "data_path": root_path + "/NTU-RGB-D-120-AGCN/xsetup/train_data_joint.npy",
      "label_path": root_path + "/NTU-RGB-D-120-AGCN/xsetup/train_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-120-AGCN/xsetup/train_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }
   
    self.test_feeder_args = {

      'data_path': root_path + "/NTU-RGB-D-120-AGCN/xsetup/val_data_joint.npy",
      'label_path': root_path + "/NTU-RGB-D-120-AGCN/xsetup/val_label.pkl",
      'num_frame_path': root_path + "/NTU-RGB-D-120-AGCN/xsetup/val_num_frame.npy",
      'l_ratio': [0.95],
      'input_size': 64
    }

class  opts_pku_part1_cross_subject():

  def __init__(self):

    # Sequence based model
    self.encoder_args = {
    "t_input_size":150,
    "s_input_size":192,
    "kernel_size":5, 
    "stride":1, 
    "padding":2, 
    "factor":2,
    "hidden_size":512,
    "num_head":4,
    "num_layer":1,
    "granularity":4,
    "encoder":"Transformer",
    "num_class":51
    }

    # feeder
    self.train_feeder_args = {
      "data_path": root_path + "/pku_part1_frame50/xsub/train_position.npy",
      "label_path": root_path + "/pku_part1_frame50/xsub/train_label.pkl",
      "num_frame_path": None ,
      'l_ratio': [0.95],
      'input_size': 64
    }
   
    self.test_feeder_args = {

      'data_path': root_path + "/pku_part1_frame50/xsub/val_position.npy",
      'label_path': root_path + "/pku_part1_frame50/xsub/val_label.pkl",
      "num_frame_path": None ,
      'l_ratio': [0.95],
      'input_size': 64
    }

class  opts_pku_part2_cross_subject():

  def __init__(self):

    # Sequence based model
    self.encoder_args = {
    "t_input_size":150,
    "s_input_size":192,
    "kernel_size":5, 
    "stride":1, 
    "padding":2, 
    "factor":2,
    "hidden_size":512,
    "num_head":4,
    "num_layer":1,
    "granularity":4,
    "encoder":"Transformer",
    "num_class":51
    }

    # feeder
    self.train_feeder_args = {
      "data_path": root_path + "/pku_part2_frame50/xsub/train_position.npy",
      "label_path": root_path + "/pku_part2_frame50/xsub/train_label.pkl",
      "num_frame_path": None ,
      'l_ratio': [0.95],
      'input_size': 64
    }
   
    self.test_feeder_args = {

      'data_path': root_path + "/pku_part2_frame50/xsub/val_position.npy",
      'label_path': root_path + "/pku_part2_frame50/xsub/val_label.pkl",
      "num_frame_path": None ,
      'l_ratio': [0.95],
      'input_size': 64
    }
