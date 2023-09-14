from Constants import FILENAMES, AGE_THRESHOLDS

import sklearn


def calculate_class_weights():
    ages = [int(x.split('_')[0]) for x in FILENAMES]
    age_distribution = {}
    for i in range(min(ages), max(ages) + 1):
        age_distribution[i] = 0

    for age in ages:
        age_distribution[age] += 1
    class_names = []

    age_classes_distribution = {}
    for i_thresh, thresh in enumerate(AGE_THRESHOLDS[:-1]):
        class_name = f'{thresh + 1} - {AGE_THRESHOLDS[i_thresh + 1]}'
        class_names.append(class_name)
        age_classes_distribution[class_name] = 0

    i = 0
    for age in age_distribution:
        if age <= AGE_THRESHOLDS[i + 1]:
            age_classes_distribution[class_names[i]] += age_distribution[age]
        else:
            i += 1

    y = [key for key in age_classes_distribution.keys() for _ in range(age_classes_distribution[key])]
    class_weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced', classes=class_names, y=y)
    return class_weights
