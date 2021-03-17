function similarity_mat = get_similarity_mat(fc, nbits)

    similarity_mat = zeros(nbits, nbits);
    
    for i=1:nbits
        parity = 2.^(i-1);
        den_i = fc.fc(fc.parity == parity);
        for j=1:nbits
            parity = 2.^(j-1);
            den_j = fc.fc(fc.parity == parity);
            if(i==j)
                continue
            end
            parity = [i,j];
            parity = sum(2.^(parity-1), 2);
            similarity_mat(i,j) = fc.fc(fc.parity == parity)+den_i+den_j;
        end
    end
    

end