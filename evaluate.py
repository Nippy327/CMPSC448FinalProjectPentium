import json

food_labels = {0: 'pizza', 1: 'sandwich', 2: 'sandwich', 3: 'sandwich', 4: 'sandwich', 5: 'sandwich', 6: 'sandwich',
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

rain_labels = {1: 'rain', 2: 'rain', 3: 'rain', 4: 'rain', 5: 'rain', 6: 'rain', 7: 'rain', 8: 'shine', 9: 'rain',
               10: 'shine', 11: 'shine', 12: 'rain', 13: 'shine', 14: 'shine', 15: 'rain', 16: 'rain', 17: 'cloudy',
               18: 'shine', 19: 'rain', 20: 'rain', 21: 'rain', 22: 'shine', 23: 'rain', 24: 'cloudy', 25: 'rain',
               26: 'shine', 27: 'cloudy', 28: 'cloudy', 29: 'cloudy', 30: 'rain', 31: 'rain', 32: 'rain', 33: 'shine',
               34: 'cloudy', 35: 'rain', 36: 'rain', 37: 'rain', 38: 'rain', 39: 'cloudy', 40: 'shine', 41: 'rain',
               42: 'rain', 43: 'shine', 44: 'rain', 45: 'shine', 46: 'rain', 47: 'rain', 48: 'cloudy', 49: 'cloudy',
               50: 'cloudy', 51: 'cloudy', 52: 'shine', 53: 'cloudy', 54: 'rain', 55: 'rain', 56: 'cloudy', 57: 'rain',
               58: 'rain', 59: 'rain', 60: 'rain', 61: 'cloudy', 62: 'cloudy', 63: 'rain', 64: 'shine', 65: 'rain',
               66: 'rain', 67: 'cloudy', 68: 'shine', 69: 'cloudy', 70: 'rain', 71: 'shine', 72: 'shine', 73: 'rain',
               74: 'cloudy', 75: 'shine', 76: 'rain', 77: 'shine', 78: 'rain', 79: 'cloudy', 80: 'shine', 81: 'shine',
               82: 'rain', 83: 'cloudy', 84: 'shine', 85: 'shine', 86: 'shine', 87: 'rain', 88: 'rain', 89: 'rain',
               90: 'shine', 91: 'cloudy', 92: 'shine', 93: 'cloudy', 94: 'rain', 95: 'cloudy', 96: 'cloudy', 97: 'rain',
               98: 'cloudy', 99: 'rain', 100: 'cloudy', 101: 'cloudy', 102: 'rain', 103: 'rain', 104: 'cloudy',
               105: 'cloudy', 106: 'rain', 107: 'rain', 108: 'shine', 109: 'cloudy', 110: 'rain', 111: 'rain',
               112: 'shine', 113: 'rain', 114: 'rain', 115: 'cloudy', 116: 'cloudy', 117: 'rain', 118: 'shine',
               119: 'cloudy', 120: 'rain', 121: 'shine', 122: 'rain', 123: 'shine', 124: 'shine', 125: 'shine',
               126: 'shine', 127: 'rain', 128: 'cloudy', 129: 'cloudy', 130: 'shine', 131: 'cloudy', 132: 'cloudy',
               133: 'shine', 134: 'cloudy', 135: 'shine', 136: 'shine', 137: 'cloudy', 138: 'cloudy', 139: 'shine',
               140: 'cloudy', 141: 'cloudy', 142: 'cloudy', 143: 'cloudy', 144: 'cloudy', 145: 'cloudy', 146: 'cloudy',
               147: 'cloudy', 148: 'shine', 149: 'shine', 150: 'cloudy', 151: 'shine', 152: 'cloudy', 153: 'cloudy',
               154: 'shine', 155: 'shine', 156: 'cloudy', 157: 'shine', 158: 'shine', 159: 'shine', 160: 'cloudy',
               161: 'shine', 162: 'shine', 163: 'shine', 164: 'cloudy', 165: 'cloudy', 166: 'cloudy', 167: 'shine',
               168: 'cloudy', 169: 'shine', 170: 'cloudy', 171: 'cloudy', 172: 'cloudy', 173: 'shine', 174: 'shine',
               175: 'shine', 176: 'shine', 177: 'shine', 178: 'shine', 179: 'shine', 180: 'shine'}


def evaluate(pred_labels: dict):
    """
    Compare the predicted labels with the actual labels and return the accuracy
    :param pred_labels: The predicted labels
    :return: Accuracy of the predicted labels
    """
    labels = rain_labels
    if len(pred_labels) != len(labels):
        raise ValueError("The length of the predicted labels is not equal to the length of the actual labels")
    if type(pred_labels) != dict:
        raise TypeError("The type of the predicted labels is not dict")

    correct = 0
    for i in range(len(labels)):
        if pred_labels[i] == labels[i]:
            correct += 1
    return correct / 115
