#ifndef GENERATE_H
#define GENERATE_H
#include <stdio.h>
#include <random>
#include <iostream>
#include <math.h>
#include <chrono>
#include "doublefann.h"
#include <string.h>
#include <chrono>

#define x_comp      0
#define y_comp      1
#define z_comp      2

#define sx_comp     3
#define sy_comp     4
#define sz_comp     5

#define ax_comp     6
#define ay_comp     7
#define az_comp     8

#define flip 1

#define PI (3.141592653589793)
#define HALF_PI (1.570796326794897)
#define OUTPUT 6 /* total output of the system, always 6 */
#define INPUT 15 /* total input of the NN. 9 without initial pose, 15 with */

#define d1  12.5   //ground to q1
#define d6  12.0   //gripper to wrist
#define a2 15.0    //q1 to q2
#define d4 19.2    //q2 to wrist

class generate
{
    public:
        generate();
        void generateDataDelta(int dataPoints, int configs);
        void generateDataDeltaFull(int dataPoints, int configs);
        void generateDataDeltaPosition(int dataPoints, int configs);
        void generateDataDeltaOrientation(int dataPoints, int configs);
        void generateData(int dataPoints);

        void forwardKinematics(double *angles, double *pos);


    private:
};

#endif // GENERATE_H
