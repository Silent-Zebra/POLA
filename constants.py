# Scheme heavily adapted from https://github.com/deepmind/pycolab/
# '@' means "wall"
# 'P' means "player" spawn point
# 'A' means apply spawn point
# '' is empty space

# Big
HARVEST_MAP_BIG = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@   P   P  P  AA P  P P AA A P  P  A P  P A  AA  A P @',
    '@  P  A  A   A A    A    AAAA A   AA    A  AAA A  A  @',
    '@ P  AA AAA  AAAA    A     A AA AAAA    AA  AA AA  P @',
    '@ A  AAA A    AA  A AAA  AA  A A A     A A AA A A    @',
    '@AAA  A A    A   AAA A AA A A AA     A A A    A   A P@',
    '@ A A  AAA  AAA   A A    A   AAA AA   AAA   AA AA AA @',
    '@  A A  AAA     A A  AAA      AAA   A A    AAA  A  P @',
    '@ P AAA  A     A AAA  A   A  AA  AA    A  AAAA     P @',
    '@  A A  AAA    AA A  AAA    A  A A AAA  A  AAA  A    @',
    '@ P  AAA A    AA  A AAA  AAA  AA A    A    A   A A  P@',
    '@    A A   AAA   A A    AA  AAA  A A   A A AA   A    @',
    '@ P  A       A   A AAA   AA  A  AA   AAAA  AA      P @',
    '@    A A     AAA   A  A    A A    A   AA A      A    @',
    '@ P  A  AAA    AA A  AAA   AA  A  A A   A  AAA  A  P @',
    '@   AAA  A      AAAA  A AA  A  A     A    AAAA       @',
    '@ P  A       A   A AAA   A  AA   A A A   A  A      P @',
    '@A  AAA  A  A  A AA A  AA  AAAA  A  A   AAAA     P   @',
    '@    A A   AAA   A A   A A A   A A  A  A A AA   A  P @',
    '@ P   AAA   A A  AAA A  AAA   A  AA    AA AA   AAA P @',
    '@ A    A     AAA  AA  AA   A A  AA  P   A       A    @',
    '@   P   P  P  A  P  AA   P  A  P A     P   P P     P @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']

# Tiny
HARVEST_MAP_TINY = [
    '@@@@@@@@@@',
    '@ P AA P @',
    '@P AAAA  @',
    '@ AAA A P@',
    '@P AAA P @',
    '@  AA    @',
    '@ P  P P @',
    '@@@@@@@@@@']

# Toy - even smaller than tiny
HARVEST_MAP_TOY = [
    '@@@@@@@',
    '@ P P @',
    '@P A P@',
    '@ AAA @',
    '@P A P@',
    '@ P P @',
    '@@@@@@@']

# single agent env from A multi-agent reinforcement learning model of
# common-pool resource appropriation
HARVEST_MAP_CPR = [
    '@@@@@@@@@@@@@@@@@@@@@@',
    '@  A   A   A   A   A @',
    '@ AAA AAA AAA AAA AAA@',
    '@  A   A   A   A   A @',
    '@                    @',
    '@    A   A   A   A   @',
    '@   AAA AAA AAA AAA  @',
    '@ P  A   A   A   A  P@',
    '@@@@@@@@@@@@@@@@@@@@@@',]

HARVEST_MAP_CPR2 = [
    '@@@@@@@@@@@@@@@@@@@@@@',
    '@P  A   A   A   A  P @',
    '@  AAA AAA AAA AAA   @',
    '@   A   A   A   A    @',
    '@                    @',
    '@  A   A   A   A   A @',
    '@ AAA AAA AAA AAA AAA@',
    '@  A   A   A   A   A @',
    '@                    @',
    '@  A   A   A   A   A @',
    '@ AAA AAA AAA AAA AAA@',
    '@  A   A   A   A   A @',
    '@                    @',
    '@    A   A   A   A   @',
    '@   AAA AAA AAA AAA  @',
    '@ P  A   A   A   A  P@',
    '@@@@@@@@@@@@@@@@@@@@@@',]


# Default
HARVEST_MAP = [
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@',
    '@ P   P      A    P AAAAA    P  A P  @',
    '@  P     A P AA    P    AAA    A  A  @',
    '@     A AAA  AAA    A    A AA AAAA   @',
    '@ A  AAA A    A  A AAA  A  A   A A   @',
    '@AAA  A A    A  AAA A  AAA        A P@',
    '@ A A  AAA  AAA  A A    A AA   AA AA @',
    '@  A A  AAA    A A  AAA    AAA  A    @',
    '@   AAA  A      AAA  A    AAAA       @',
    '@ P  A       A  A AAA    A  A      P @',
    '@A  AAA  A  A  AAA A    AAAA     P   @',
    '@    A A   AAA  A A      A AA   A  P @',
    '@     AAA   A A  AAA      AA   AAA P @',
    '@ A    A     AAA  A  P          A    @',
    '@       P     A         P  P P     P @',
    '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@']


CLEANUP_MAP_SMALL = [
    '@@@@@@@@@@@@@@@@@@',
    '@RRRRRR     BBBBB@',
    '@HHHHHH   P  BBBB@',
    '@RRRRR  P    BBBB@',
    '@RRRRR    P BBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@RRRRR   P P BBBB@',
    '@HHHHH   P  BBBBB@',
    '@RRRRRR    P BBBB@',
    '@HHHHHH P   BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@@@@@@@@@@@@@@@@@@']


CLEANUP_MAP = [
    '@@@@@@@@@@@@@@@@@@',
    '@RRRRRR     BBBBB@',
    '@HHHHHH      BBBB@',
    '@RRRRRR     BBBBB@',
    '@RRRRR  P    BBBB@',
    '@RRRRR    P BBBBB@',
    '@HHHHH       BBBB@',
    '@RRRRR      BBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@HHHHHHSSSSSSBBBB@',
    '@RRRRR   P P BBBB@',
    '@HHHHH   P  BBBBB@',
    '@RRRRRR    P BBBB@',
    '@HHHHHH P   BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH    P  BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH  P P BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHHH      BBBBB@',
    '@RRRRR       BBBB@',
    '@HHHH       BBBBB@',
    '@@@@@@@@@@@@@@@@@@']
