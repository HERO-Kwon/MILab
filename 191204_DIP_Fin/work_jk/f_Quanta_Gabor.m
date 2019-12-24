function QGF = f_Quanta_Gabor(Gabor,np,nm)

A_p = max(max(Gabor));
A_m = min(min(Gabor));



for k=1:np+1
    g_p(k) = A_p/(2*np+1)*(2*(k-1));
end

for k=1:nm+1
    g_m(k) = A_m/(2*nm+1)*(2*(k-1));
end

step_p = g_p(2)-g_p(1);         % step size Plus
step_m = g_m(2)-g_m(1);         % step size Minus

level_p(1) = 0;
level_m(1) = 0;

level_p(2) = step_p * (1/2);
level_m(2) = step_m * (1/2);

if np>=2
    for k=3:np+1
        level_p(k) = level_p(k-1)+step_p*(k-2);
    end
end

if nm>=2
    for k=3:nm+1
        level_m(k) = level_m(k-1)+step_m*(k-2);        
    end
end

for y=1:size(Gabor,1)
    for x=1:size(Gabor,2)
        val = Gabor(y,x);
        if(val>=0)
            for k=1:np
                if( (val>=level_p(k)) && (val<level_p(k+1)) )
                    QGF(y,x) = g_p(k);
                elseif( val>=level_p(k+1) )
                    QGF(y,x) = g_p(k+1);
                end
            end
        elseif(val<0)
            for k=1:nm
                if( (val<=level_m(k)) && (val>level_m(k+1)) )
                    QGF(y,x) = g_m(k);
                elseif(val <= level_m(k+1))
                   QGF(y,x) = g_m(k+1);                     
                end
            end
        end   
    end
end


