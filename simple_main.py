import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from brdf_trainer import BRDFTrainer
from model.brdf import MLPPBRBRDF

@hydra.main(config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    material = MLPPBRBRDF()
    model = BRDFTrainer(cfg, material)
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        default_root_dir=cfg.exp_output_root_path,
        accelerator='gpu',
        devices=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(monitor='val/loss', save_top_k=1, save_last=True),
            pl.callbacks.LearningRateMonitor()
        ]
    )
    trainer.fit(model)
    trainer.test(model)

if __name__ == '__main__':
    main()
