import json

labels = {0: 'pizza', 1: 'sandwich', 2: 'sandwich', 3: 'sandwich', 4: 'sandwich', 5: 'sandwich', 6: 'sandwich',
          7: 'sandwich', 8: 'sandwich', 9: 'sandwich', 10: 'sandwich', 11: 'sandwich', 12: 'sandwich', 13: 'sandwich',
          14: 'sandwich', 15: 'sandwich', 16: 'sandwich', 17: 'sandwich', 18: 'sandwich', 19: 'sandwich',
          20: 'sandwich', 21: 'sandwich', 22: 'sandwich', 23: 'sandwich', 24: 'sandwich', 25: 'sandwich',
          26: 'sandwich', 27: 'sandwich', 28: 'sandwich', 29: 'sandwich', 30: 'sandwich', 31: 'sandwich',
          32: 'sandwich', 33: 'sandwich', 34: 'sandwich', 35: 'sandwich', 36: 'sandwich', 37: 'sandwich',
          38: 'sandwich', 39: 'sandwich', 40: 'sandwich', 41: 'ice_cream', 42: 'ice_cream', 43: 'ice_cream',
          44: 'ice_cream', 45: 'pizza', 46: 'ice_cream', 47: 'pizza', 48: 'ice_cream', 49: 'pizza', 50: 'ice_cream',
          51: 'pizza', 52: 'ice_cream', 53: 'pizza', 54: 'ice_cream', 55: 'pizza', 56: 'ice_cream', 57: 'pizza',
          58: 'ice_cream', 59: 'pizza', 60: 'pizza', 61: 'pizza', 62: 'pizza', 63: 'pizza', 64: 'ice_cream',
          65: 'pizza', 66: 'pizza', 67: 'pizza', 68: 'pizza', 69: 'ice_cream', 70: 'pizza', 71: 'ice_cream',
          72: 'ice_cream', 73: 'ice_cream', 74: 'ice_cream', 75: 'ice_cream', 76: 'ice_cream', 77: 'pizza', 78: 'pizza',
          79: 'pizza', 80: 'pizza', 81: 'pizza', 82: 'pizza', 83: 'pizza', 84: 'ice_cream', 85: 'ice_cream',
          86: 'pizza', 87: 'ice_cream', 88: 'ice_cream', 89: 'pizza', 90: 'pizza', 91: 'ice_cream', 92: 'ice_cream',
          93: 'pizza', 94: 'ice_cream', 95: 'ice_cream', 96: 'ice_cream', 97: 'ice_cream', 98: 'pizza', 99: 'ice_cream',
          100: 'pizza', 101: 'pizza', 102: 'pizza', 103: 'pizza', 104: 'pizza', 105: 'ice_cream', 106: 'ice_cream',
          107: 'ice_cream', 108: 'ice_cream', 109: 'ice_cream', 110: 'ice_cream', 111: 'ice_cream', 112: 'ice_cream',
          113: 'pizza', 114: 'ice_cream', 115: 'pizza'}


def evaluate(pred_labels: dict):
    """
    Compare the predicted labels with the actual labels and return the accuracy
    :param pred_labels: The predicted labels
    :return: Accuracy of the predicted labels
    """
    if len(pred_labels) != len(labels):
        raise ValueError("The length of the predicted labels is not equal to the length of the actual labels")
    if type(pred_labels) != dict:
        raise TypeError("The type of the predicted labels is not dict")

    correct = 0
    for i in range(len(labels)):
        if pred_labels[i] == labels[i]:
            correct += 1
    return correct / 115

