#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " N\n";
    return 1;
  }

  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(argv[1], &end, 10);
  if (errno != 0 || end == argv[1] || *end != '\0' || parsed < 0 ||
      parsed > INT_MAX) {
    std::cerr << "N must be a non-negative integer.\n";
    return 1;
  }

  const int n = static_cast<int>(parsed);

  for (int i = 0;; ++i) {
    std::printf("%d%c", i, i == n ? '\n' : ' ');
    if (i == n) {
      break;
    }
  }

  for (int i = n;; --i) {
    std::cout << i << (i == 0 ? '\n' : ' ');
    if (i == 0) {
      break;
    }
  }

  return 0;
}
