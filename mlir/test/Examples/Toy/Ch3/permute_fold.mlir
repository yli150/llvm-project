module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00, 1.000000e+00, 2.000000e+00]> : tensor<4xf64>
    %1 = toy.reshape(%0 : tensor<4xf64>) to tensor<2x2xf64>
    %2 = toy.permute(%1 : tensor<2x2xf64>) { perm = [0, 1] } to tensor<2x2xf64>
    toy.print %2 : tensor<2x2xf64>
    toy.return
  }
}