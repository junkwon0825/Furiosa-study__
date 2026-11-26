import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

plt.plot([1, 2, 3], [10, 20, 30])

plt.title('한글 테스트')
plt.xlabel('에포크')
plt.ylabel('손실값')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
plt.show()