cmake_minimum_required(VERSION 2.8)
find_package( OpenCV REQUIRED )

set(source ./../liblinear/blas/blas.h ./../liblinear/blas/blasp.h ./../liblinear/blas/daxpy.c ./../liblinear/blas/ddot.c ./../liblinear/blas/dnrm2.c ./../liblinear/blas/dscal.c ./../liblinear/tron.cpp ./../liblinear/tron.h ./../liblinear/linear.h ./../liblinear/linear.cpp
./../headers.h ./../utils.h ./../randomforest.h ./../regressor.h ./../utils.cpp ./../randomforest.cpp ./../regressor.cpp ./../main.cpp)

add_executable(application ${source})
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fopenmp")
target_link_libraries(application ${OpenCV_LIBS}) 

