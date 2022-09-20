# Syntax:
# (argument option pairs [that trigger the migration]): migration function that takes list of args
from util.migrations import remove_arg, replace_arg, change_model_to

RULES = {
    (("--no-mean-subtraction",),): remove_arg("--no-mean-subtraction"),
    (("--no-normalization",),): remove_arg("--no-normalization"),
    (("--model", "meliusnet25_4"),): replace_arg(("--model", "meliusnet25_4"),
                                                ("--model", "meliusnet_a")),
    (("--model", "meliusnet29_2"),): replace_arg(("--model", "meliusnet29_2"),
                                                ("--model", "meliusnet_b")),
    (("--improvement-block-type", "last"),): change_model_to("meliusnet", "--improvement-block-type"),
    (("--improvement-block-type", "all"),): change_model_to("naivenet", "--improvement-block-type"),
    (("--block-config", "6,6,6,5"), ("--model", "densenet_flex")): change_model_to("densenet", ""),
    (("--block-config", "6,8,12,6"), ("--model", "densenet_flex")): change_model_to("densenet", ""),
    (("--initial-layers", "deepstem_grouped"),): replace_arg(("--initial-layers", "deepstem_grouped"),
                                                            ("--initial-layers", "grouped_stem")),
}
