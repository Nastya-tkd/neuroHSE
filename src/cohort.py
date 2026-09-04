"""
The valid subject cohort for the calibrated-fMRI experiment (excludes the
separate dp-/r- prefixed control/replication-study subjects, which use a
different design). Derived from ds004873/participants.tsv (recovered via
S3 version history, see scripts/download_real_labels.py) by keeping every
sub-pXXX row with a non-empty Hct value and no "EXCLUDED" comment. Matches
the paper's own "N40" cohort size (seen in derivative filenames like
N40_cond-control_space-MNI152_median_cbf.nii.gz).

Excluded (per participants.tsv comments): sub-p022 (behavioral problems),
sub-p024/sub-p025 (contrast agent did not reach participant),
sub-p042 (CBF maps unilateral), sub-p053 (T2* maps unilateral),
sub-p056 (movement artifacts/ringing), sub-p062 (susceptibility artifacts).
"""

ALL_SUBJECTS = [
    "sub-p019", "sub-p020", "sub-p021", "sub-p023", "sub-p026", "sub-p027",
    "sub-p028", "sub-p029", "sub-p030", "sub-p031", "sub-p032", "sub-p033",
    "sub-p034", "sub-p035", "sub-p036", "sub-p037", "sub-p038", "sub-p039",
    "sub-p040", "sub-p043", "sub-p044", "sub-p046", "sub-p047", "sub-p048",
    "sub-p049", "sub-p050", "sub-p051", "sub-p052", "sub-p054", "sub-p055",
    "sub-p058", "sub-p059", "sub-p060", "sub-p061", "sub-p063", "sub-p064",
    "sub-p065", "sub-p066", "sub-p067", "sub-p068",
]
