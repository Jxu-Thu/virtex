from pytorch_lightning.utilities.cli import LightningCLI
from modules.MMC import Mixed_Moco
from datamodules.coco2017_datamodule import CocoCaptionKarpathyDataModule
class My_CLI(LightningCLI):
    def __init__(
        self,
        is_test: bool = False,
        **kwargs,
    ) -> None:
        """Add a parameter is_test to determine the mode
        Args:
            is_test: whether only testing
            kwargs: Original parameters of LightningCLI
        """
        self.is_test = is_test
        super().__init__(**kwargs)

    def fit(self):
        """Runs fit of the instantiated trainer class and prepared fit keyword arguments"""
        if self.is_test:
            pass
        else:
            self.trainer.fit(**self.fit_kwargs)
    def after_fit(self):
        """Runs testing"""
        if self.is_test:
            self.trainer.test(self.model, datamodule=self.datamodule)
        else:
            pass

def main():
#     print('hh')
    cli = My_CLI(model_class= Mixed_Moco, datamodule_class = CocoCaptionKarpathyDataModule, is_test = False, seed_everything_default=1234)

if __name__ == '__main__':
    main()
    # print('hh')