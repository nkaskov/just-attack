"""Module for working with vectors over GF(2)."""


class Vector():
    """Binary vector abstraction."""

    def __init__(self, value=None, length=None):
        """Create new vector of size.

        :param: int `value` - integer representation of bit vector
        :param: int `length` - length of the vector
        """
        self._len = 0
        self._vector = 0
        if length:
            if not isinstance(length, int):
                raise TypeError(
                    'expected `length` is integer, got {}'
                    ''.format(type(length)))
            if length < 0:
                raise ValueError(
                    'expected `length` is non negative number,'
                    'got {} < 0'.format(length))
            self._len = length
        if value:
            if not isinstance(value, int):
                raise TypeError(
                    'expected  `value` is integer, got {}'.format(type(value)))
            if value < 0:
                raise ValueError(
                    'expected `value` is non negative integer, '
                    'got {} < 0'.format(value))
            if length > 0:
                self._vector = value & ((1 << length) - 1)
            else:
                self._vector = value
                self._len = len(bin(self._vector[2:]))

    @property
    def value(self):
        """Return raw value of vector."""
        return self._vector

    @property
    def hamming_weight(self):
        """Return Hamming weight of vector.

        Hamming weight = a count of ones.
        """
        hamming_weight = 0
        for i in range(len(self)):
            if self._vector & (1 << (len(self) - i - 1)):
                hamming_weight += 1
        return hamming_weight

    @property
    def support(self):
        """Return list of one's positions of vector.

        Example:
            001101.support = [2, 3, 5]
        """
        return list(self.iter_support())

    @property
    def support_supplement(self):
        """Return list of zeroes positions of vector.

        Example:
            001101.support = [0, 1, 4]
        """
        return list(self.iter_support_supplement())

    def __len__(self):
        """Return length of vector."""
        return self._len

    def set_length(self, length):
        """Change length of a vector.

        10011.set_length(7) -> 0010011
        10011.set_length(3) -> 011
        """
        if not isinstance(length, int):
            raise TypeError('expected `length` is integer, not {}'
                            ''.format(type(length)))
        if length < 0:
            raise ValueError('expected `length` is greater than 0, '
                             'but {} < 0'.format(length))
        self._vector = self._vector & ((1 << length) - 1)
        self._len = length
        return self

    def resize(self, delta_length):
        """Change size of vector by 'delta_length'."""
        if not isinstance(delta_length, int):
            raise TypeError(
                'expected `delta_length` is integer, got {}'
                ''.format(type(delta_length)))
        return self.set_length(self._len + delta_length)

    def copy(self):
        """Return copy of vector."""
        return self.__class__(self.value, len(self))

    def iter_support(self):
        """Return iterator over one's positions of vector."""
        for i in range(len(self)):
            if self._vector & (1 << (len(self) - i - 1)):
                yield i

    def iter_support_supplement(self):
        """Return iterator over zeroes positions of vector."""
        for i in range(len(self)):
            if not self._vector & (1 << (len(self) - i - 1)):
                yield i

    def to_str(self, zerofiller=None, onefiller=None):
        """Return string representation of vector."""
        if not self:
            return ''
        str_vec = bin(self._vector)[2:].zfill(len(self))
        if onefiller:
            str_vec = str_vec.replace("1", onefiller)
        if zerofiller:
            str_vec = str_vec.replace("0", zerofiller)
        return str_vec

    def concatenate(self, other):
        """Concatenate of two vectors."""
        self._vector = (self.value << (len(other))) ^ other.value
        self._len = len(self) + len(other)
        return self

    def __bool__(self):
        """Return True if and only if len > 0."""
        if self._len == 0:
            return False
        return True

    def __repr__(self):
        """Return representation of Vector class as string."""
        rep = 'Vector(len={}, [{vector}])'
        return rep.format(len(self), vector=str(self))

    def __str__(self):
        """Return representation of Vector as string to print."""
        return self.to_str()

    def __setitem__(self, index, value):
        """Set item of vector: vector[index] = value."""
        if not isinstance(index, int):
            raise TypeError("`index` must be integer not "
                            "`{}`".format(type(index)))

        if isinstance(value, str) and value == '0':
            value = False
        else:
            value = bool(value)
        index = self._len - (index % self._len) - 1
        bit = self._vector & (1 << index)
        self._vector ^= bit  # set bit in zero
        if value:
            self._vector ^= (1 << index)  # set value

    def __getitem__(self, index):
        """Return vector[index].

        If `index` is integer then function returns integer 0 or 1.
        If `index` is slice then function returns instance of Vector.

        """
        if isinstance(index, int):
            index = self._len - (index % self._len) - 1
            if self._vector & (1 << index):
                return 1
            return 0
        # index is slice
        try:
            vec_len = 0
            vec_value = 0
            for i in range(*index.indices(self._len)):
                vec_value <<= 1
                if self._vector & (1 << (self._len - i - 1)):
                    vec_value ^= 1
                vec_len += 1
        except AttributeError:
            raise TypeError(
                '`index` must be integer or slice, not '
                '`{}`'.format(type(index)))
        if not vec_len:
            return None
        return Vector(value=vec_value, length=vec_len)

    def __eq__(self, other):
        """Return True if self == other, else return False."""
        if not isinstance(other, self.__class__):
            raise ValueError("expected other is `Vector`, not {}"
                             "".format(type(other)))
        return len(self) == len(other) and self._vector == other.value


    def __hash__(self):
        return self.to_str()


    def __ne__(self, other):
        """Return True if self != other, else return False."""
        return not self == other

    def __imul__(self, other):
        """Bitwise vector multiplication.

        self = self * other and return self
        """
        if not isinstance(other, self.__class__):
            raise TypeError("expected `Vector` object, not {}"
                            "".format(type(other)))

        self._vector &= other.value
        return self

    def __mul__(self, other):
        """Bitwise vector multiplication.

        return self * other
        """
        mul = self.__class__(self._vector, len(self))
        mul *= other
        return mul

    def __iand__(self, other):
        """Bitwise AND of vectors.

        self = self & other and return self
        """
        self *= other
        return self

    def __and__(self, other):
        """Bitwise AND of vectors.

        return self & other
        """
        return self * other

    def __iadd__(self, other):
        """Bitwise vector addition (xor).

        self = self + other and return self
        """
        if not isinstance(other, self.__class__):
            raise TypeError("expected `Vector` object, not {}"
                            "".format(type(other)))

        self._vector = self._vector ^ other.value
        return self

    def __add__(self, other):
        """Bitwise vector addition (xor).

        return self + other
        """
        summa = self.__class__(self._vector, len(self))
        summa += other
        return summa

    def __ixor__(self, other):
        """Bitwise xor of vectors.

        self = self ^ other and return self
        """
        self += other
        return self

    def __xor__(self, other):
        """Bitwise xor of vectors.

        return self ^ other
        """
        return self + other

    def __ior__(self, other):
        """Bitwise OR of vectors.

        self = self | other and return self
        """
        if not isinstance(other, self.__class__):
            raise TypeError("expected `Vector` object, not {}"
                            "".format(type(other)))

        self._vector = self._vector | other.value
        return self

    def __or__(self, other):
        """Bitwise OR of vectors.

        return self | other
        """
        or_vector = self.__class__(self._vector, len(self))
        or_vector |= other
        return or_vector

    def bitwise_not(self):
        """Bitwise NOT of vector.

        return not(self)
        """
        self._vector = (1 << self._len) - 1 - self._vector
        return self

    def __ilshift__(self, pos):
        """Non cyclic left shift of vector by `pos`.

        self <<= pos and return self
        """
        self._vector = (self._vector << pos) & ((1 << self._len) - 1)
        return self

    def __irshift__(self, pos):
        """Non cyclic right shift of vector by `pos`.

        self >>= pos and return self
        """
        self._vector >>= pos
        return self

    def __lshift__(self, pos):
        """Non cyclic left shift of vector by `pos`.

        return self << pos
        """
        vec = self.copy()
        vec <<= pos
        return vec

    def __rshift__(self, pos):
        """Non cyclic right shift of vector by `pos`.

        return self >> pos
        """
        vec = self.copy()
        vec >>= pos
        return vec

    def __iter__(self):
        """Iterate over elements of vector."""
        for i in range(len(self)):
            yield int(bool(self._vector & (1 << (len(self) - i - 1))))

    def __int__(self):
        """Convert vector object to integer."""
        return self._vector

    def to_latex_str(self):
        """Return string representation of vector to insert in LaTeX document.

        Example:
            '0011101' -> '0&0&1&1&1&0&1'
        """
        latex = str(self).replace('1', '1&').replace('0', '0&')
        return latex[:-1] if latex else ''


def bitwise_not(vector):
    """Return bitwise NOT of vector."""
    bt_vector = vector.copy()
    bt_vector.bitwise_not()
    return bt_vector


def from_support(length, support=None):
    """Return Vector by set of ones."""
    if not isinstance(length, int):
        raise TypeError(
            'expected `length` is integer, but got {}'
            ''.format(type(length)))
    if length < 0:
        raise ValueError(
            'expected `length` is not less then 0, but got'
            ' {} < 0'.format(length))
    if not support:
        return Vector(0, length)
    value = 0
    try:
        for i in support:
            try:
                value ^= (1 << (length - 1 - (i % length)))
            except TypeError:
                raise ValueError(
                    'except index is integer, but got'
                    ' {} is {}'.format(i, type(i)))
    except TypeError:
        raise TypeError('expected `support` is iterable')
    return Vector(value, length)


def from_support_supplement(length, support_supplement=None):
    """Return Vector by set of zeroes."""
    try:
        if support_supplement:
            support = (i for i in range(length) if i not in support_supplement)
        else:
            support = range(length)
    except TypeError:
        raise TypeError(
            'expected `length` is string, but got'
            ' {}'.format(type(length)))
    return from_support(length, support)


def from_string(value, zerofillers=None, onefillers=None):
    """Return vector from string.

    :param: str `values` - string representation of binary vector;
    :param: list or string `zerofillers` - possible fillers of '0';
    :param: list or string `onefillers` - possible fillers of '1'.

    Example:
    ('110**|0_1', zerofillers=['*', '_'], onefillers='|') ->    110001001
    """
    if not isinstance(value, str):
        raise TypeError(
            'expected `value` is string, got {}'.format(type(value)))
    value = __clear_str_from_fillers(value, '0', zerofillers)
    value = __clear_str_from_fillers(value, '1', onefillers)
    try:
        vector = int(value, 2)
    except ValueError:
        if value:
            raise ValueError(
                'cannot convert string `{}` to binary vector'
                ''.format(value))
        vector = 0
    return Vector(vector, len(value))


def from_iterable(value, zerofillers=None, onefillers=None):
    """Return vector from string.

    :param: `value` - representation of binary vector as iterable;
    :param: list or string `zerofillers` - possible fillers of '0';
    :param: list or string `onefillers` - possible fillers of '1'.

    Example:
    (('110**|0_1', True, False, 0, 1 ,10, [1, 2] , []),
        zerofillers=['*', '_'], onefillers='|') -> 1100010011001110
    """
    vector = ''
    try:
        for i in value:
            if isinstance(i, str):
                vector += i
            elif i:
                vector += '1'
            else:
                vector += '0'
    except TypeError:
        raise TypeError(
            'expected `value` has any iterable type, got {}'
            ''.format(type(value)))
    return from_string(vector,
                       zerofillers=zerofillers,
                       onefillers=onefillers)


def hamming_distance(vector_a, vector_b):
    """Return Hamming distance between vectors."""
    return (vector_a + vector_b).hamming_weight


def scalar_product(vector_a, vector_b):
    """Return scalar product of two vectors."""
    return (vector_a * vector_b).hamming_weight % 2


def concatenate(first, second):
    """Concatenate of two vectors."""
    return Vector(
        (first.value << (len(second))) ^ second.value,
        len(first) + len(second))


def __clear_str_from_fillers(string, symbol, filler):
    """Replace all occurrence of `filler` in `string` with `symbol`."""
    if not filler:
        return string
    try:
        for i in filler:
            try:
                string = string.replace(i, symbol)
            except TypeError:
                raise TypeError(
                    'expected filler is string, '
                    'but got `{}` is {}'.format(i, type(i)))
    except TypeError:
        try:
            string = string.replace(filler, symbol)
        except TypeError:
            raise TypeError(
                'expected filler is string, '
                'but got `{}` is {}'.format(filler, type(filler)))
    return string
