
def get_pretraining_set_intra(opts):

    from feeder.feeder_pretraining import Feeder
    training_data = Feeder(**opts.train_feeder_args)

    return training_data


def get_finetune_training_set(opts):

    from feeder.feeder_downstream import Feeder

    data = Feeder(**opts.train_feeder_args)

    return data

def get_finetune_validation_set(opts):

    from feeder.feeder_downstream import Feeder
    data = Feeder(**opts.test_feeder_args)

    return data

def get_semi_training_set(opts):

    from feeder.feeder_semi import Feeder

    data = Feeder(**opts.train_feeder_args)

    return data
