"""Actor-wise cross-validation and matched-protocol backbone comparison for
speech + face emotion recognition on RAVDESS.

Modules:
    data      RAVDESS loading, clip table, audio preprocessing, blendshape alignment
    folds     actor-wise fold assignment (6 x 4 test actors, 2M/2F)
    models    pipeline systems (W2V2-WLP, W2V2-LoRA, E2V-LP) and the shared matched-protocol probe
    train     seeding, training loop, prediction
    fer       blendshape summary features and the XGBRF classifier
    fusion    temperature scaling and calibrated sum fusion
    runner    per-(fold, seed, system) runs with resumable on-disk outputs
    analysis  metrics, actor bootstrap, paired tests, figures, RESULTS_CV.md
    testing   offline stand-ins (synthetic data, tiny random backbones) for smoke tests
"""
