module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64>


    %10 = toy.constant dense<[2.000000e+00]> : tensor<1xf64>
    %11 = toy.reshape(%10 : tensor<1xf64>) to tensor<1xf64>
    toy.print %11 : tensor<1xf64>

    %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<1x2x1xf64>
    toy.print %2 : tensor<1x2x1xf64>
    toy.return
  }
}