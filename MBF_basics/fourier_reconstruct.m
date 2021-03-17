function y = fourier_reconstruct(fourier_coeff, x, nbits)
    parity = de2bi(fourier_coeff.parity, nbits);
    parity = (-1).^sum(x.*parity,2);
    y = sum(fourier_coeff.fc.*parity);
end