def partitionDisjoint(A) -> int:
    lmax = A[0]
    amax = A[0]
    l = 0
    for n in range(1, len(A)):
        print(lmax, amax, l, n)
        if A[n] > lmax:
            amax = max(A[n], amax)
        elif A[n] < lmax:
            l = n
            lmax = amax
    return l + 1

print(partitionDisjoint([5,0,3,8,6]))