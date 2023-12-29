module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<6xf64>
    %1 = toy.reshape(%0 : tensor<6xf64>) to tensor<2x3xf64>
    %2 = toy.permute(%1 : tensor<2x3xf64>) { perm = [1, 0] } to tensor<3x2xf64>
    toy.print %2 : tensor<3x2xf64>
    toy.return
  }
}