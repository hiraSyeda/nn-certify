from train_gloro import script

script(
    dataset='mnist',
    epsilon=0.02,
    epsilon_schedule='[0.01]-log-[50%:1.1]',
    epochs=10,
    loss='sparse_trades_kl.1.5',
    augmentation='none')
