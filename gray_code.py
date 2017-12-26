"""
Gray code package by A.A. Barashkov, MIREA
"""


def int2gray(a, bit_size):
    """
    Convert an integer to Gray code
    :param a: integer
    :param bit_size: length of array with Gray code
    :return: array with Gray code
    """
    bit_a = [int(digit) for digit in bin(a)[2:]]
    assert len(bit_a) <= bit_size
    bit_a = list(map(bool, bit_a))
    bit_a = [False]*(bit_size-len(bit_a)) + bit_a
    return bin2gray(bit_a)


def bin2gray(bit_a):
    """
    Convert an array with binary code to Gray code
    :param bit_a: array with binary code
    :return: array with Gray code
    """
    bit_a_shift = [bit_a[-1]] + bit_a[:-1]
    result = [b != bs for b, bs in zip(bit_a, bit_a_shift)]
    result = list(map(int, result))
    return result


def gray2int(gray_code):
    """
    Convert an array with Gray code to an integer
    :param gray_code: array with Gray code
    :return: integer
    """
    binary = gray2bin(gray_code)
    base = 1
    result = 0
    binary.reverse()
    for bit in binary:
        result += bit * base
        base *= 2
    return result


def gray2bin(gray_code):
    """
    Convert an array with Gray code to binary code
    :param gray_code: array with Gray code
    :return: array with binary code
    """
    gray_code = list(map(bool, gray_code))
    binary = gray_code.copy()
    shifted = gray_code
    for _ in range(len(gray_code) - 1):
        shifted = [False] + shifted[:-1]
        binary = [b != bs for b, bs in zip(binary, shifted)]
    binary = list(map(int, binary))
    return binary
