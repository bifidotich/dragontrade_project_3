import datetime
import config
import utils
import factory

if __name__ == '__main__':
    path_log = f'{config.PATH_LOGS}/run_' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.txt'
    logger = utils.files.ConsoleLogger(filename=path_log)
    utils.files.mother_iam_coder()

    # factory.load_symbols()
    factory.clear_temp()
    factory.prepare_date_out()
    factory.view_results_date_out()
    factory.prepare_data_train()

    factory.train_cnn(model_type='Lera', new=True, epochs=1000)  # Wyvern, Kite, Eagle, Peacock, Lera
    # factory.train_forest()

    factory.copy_in_production()
    utils.files.backup_directory(config.PATH_TEMP, config.PATH_AUTO_BACKUP, action='copy')

    logger.close()
