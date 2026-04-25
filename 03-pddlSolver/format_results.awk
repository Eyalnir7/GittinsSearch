BEGIN {
    FS=",";
}
{
    if (NR > 1) {
        num = $7 + $8;
        title = $4 " " $5;
        if ($4 == 9999) {
            title = "no minimal conflict";
        } else {
            if ($5 == " false") {
                title = "full binary search";
            } else {
                title = "MDP";
            }
        }
        
        data[$1, title, $2, $3] = num;    
        goals[$1] = $1;
        titles[title] = title;
        Ns[$2] = $2;
        discounts[$3] = $3;
    }
}
END {
    for (g in goals) {
        print "";
        print "";
        print g;
        for (title in titles) {
            print "";
            print title;            
            printf("discount/N, ");
            for (N in Ns) {
                printf("%d, ", N);
            }
            print "";
            for (discount in discounts) {
                printf("%0.1f, ", discount);
                for (N in Ns) {
                    printf("%d, ", data[g, title, N, discount]);
                }
                print "";
            }
        }        
    }
}