# g++ -std=c++14 \
#     -I/path/to/your/project \
#     -I/opt/homebrew/include \
#     -L/opt/homebrew/lib \
#     main.cpp sampler/polyagamma.cpp io/csv.cpp distributions/distributions.cpp \
#     -ldlib \
#     -o main

g++ -std=c++14 \
    -I/path/to/your/project \
    -I/opt/homebrew/include \
    -L/opt/homebrew/lib \
    main.cpp io/csv.cpp distributions/distributions.cpp sampler/polyagamma.cpp ibp/ibp.cpp ibp/ibpfactor.cpp \
    -ldlib \
    -framework Accelerate \
    -o gibbs_sampler
