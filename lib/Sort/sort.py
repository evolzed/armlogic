

class Sort:

    list = list()
    # 比较每个子元素中的第m个元素

    def bubble_sort(self, list, m, n):

        length = len(list)

        for index in range(length):

            for j in range(1, length - index):

                if list[j - 1][m] > list[j][m]:

                    list[j - 1][m], list[j][m] = list[j][m], list[j - 1][m]

        return list


if __name__ == '__main__':
    0
