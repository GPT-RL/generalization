import pandas as pd

df = pd.read_csv("~/generalization/data.csv", sep="\t")

common_features = df.Feature.value_counts().index[:20]
df1 = df[df.Feature.isin(common_features)]

common_features = df.Feature.value_counts().gt(20)
common_features = common_features.index[common_features.values]
df = df[df.Feature.isin(common_features)]
CONCEPT = "Concept"

"""
Principles:
1. Remove features that occur fewer than five times
2. Each Concept's features should have membership
in both sets (to avoid "none-of-the-above", which could be confusing).
3. Each Concept's features should have no more
than one member in either set (to avoid ambiguous characterizations, as in the case of multi-color objects).
4. Feature sets should share a theme.
5. Avoid subjective features like "is_large".
"""

has_color = df[
    df.Feature.isin(
        [
            "is_brown",
            # "is_white",
            "is_black",
            "is_green",
            # "is_yellow",
            "is_red",
            # "is_orange",
            "is_grey",
        ]
    )
]
has_color = has_color.groupby(CONCEPT).filter(lambda x: len(x) == 1)
made_of_material = df[df.Feature.isin(["an_animal", "made_of_metal", "made_of_wood"])]
made_of_material = made_of_material.groupby(CONCEPT).filter(lambda x: len(x) == 1)
combined = (
    pd.concat([has_color, made_of_material])
    .groupby(CONCEPT)
    .filter(lambda x: len(x) == 2)
)
breakpoint()
# food = df.groupby(CONCEPT).filter(lambda x: (x.Feature == "is_edible").any())
# food = food[food.Feature.isin(["an_animal", "a_fruit", "a_vegetable"])]
# food = food.groupby(CONCEPT).filter(lambda x: len(x) == 1)
# combined = pd.concat([has_color, food]).groupby(CONCEPT).filter(lambda x: len(x) == 2)
#
# breakpoint()

# is_color = is_color.groupby(CONCEPT).filter(lambda x: len(x) == 1)
# data = pd.concat([made_of, is_color])
# data = data.groupby(CONCEPT).filter(lambda x: len(x) == 2)
# filtered = df[
#     df.Feature.isin(
#         [
#             "made_of_metal",
#             # "an_animal",
#             "made_of_wood",
#             # "is_edible",
#             # "is_round",
#             # "different_colours",
#             # "is_brown",
#             # "made_of_plastic",
#             # "is_white",
#             # "is_black",
#             # "has_4_legs",
#             # "is_green",
#             # "has_legs",
#             # "has_wings",
#             # "has_a_handle",
#             # "a_mammal",
#         ]
#     )
# ]
# breakpoint()
#
# groups = filtered.groupby(CONCEPT)
# features = np.expand_dims(common_features, 0)
#
#
# def f(x):
#     _features = np.expand_dims(x.Feature.values, 1)
#     return np.sum(_features == features, axis=0)
#
#
# feature_bits = groups.apply(f)
# feature_array = np.stack(feature_bits.values.tolist())
# has_both_features = np.expand_dims(feature_array, axis=1) & np.expand_dims(
#     feature_array, axis=2
# )
# has_both_features = has_both_features.any(0)
# has_either_feature = np.expand_dims(feature_array, axis=1) | np.expand_dims(
#     feature_array, axis=2
# )
# has_either_feature = has_either_feature.sum(0)
# exclusive_pairs_coverage = has_either_feature * ~has_both_features
# ordering = exclusive_pairs_coverage.flatten().argsort()
# f1, f2 = np.unravel_index((ordering), exclusive_pairs_coverage.shape)
#
# f1 = common_features.values[f1]
# f2 = common_features.values[f2]
# for i, (f1, f2) in enumerate(zip(f1, f2)):
#     print(f1, f2)
#
# # print(list(exclusive_pairs()))
# # breakpoint()
#
# # feature_counts = df.Feature.value_counts()
#
# breakpoint()
# filtered = filtered.groupby(CONCEPT).filter(f)
# groups = filtered.groupby(CONCEPT)
# breakpoint()
