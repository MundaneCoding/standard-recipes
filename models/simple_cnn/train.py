from commons.launch_tools import StandardTrainer, StandardArgParser
from models.simple_cnn.model import SimpleClassifier
from commons.launch_tools import StandardTrainer, StandardArgParser

if __name__ == '__main__':
    parser = StandardArgParser()
    args = parser.parse_args()

    model = SimpleClassifier()

    data_dir='/mnt/data/data_dir' # actual data folder
    index_dir='/mnt/data/index' # train, val, test index

    datamodule = SimpleDatamodule(data_dir, index_dir, 
                                    batch_size=args.batch_size)
    datamodule.setup(stage='')

    trainer = StandardTrainer(sargs=args)
    trainer.fit(model, datamodule)