#include "generate.h"


generate::generate()
{

}
/* takes in array with range [-1,1] and outputs array with range [-1,1]. Orientation is given by the approach and sliding direction of the gripper. technically 3DOF but 6 outputs */
void generate::forwardKinematics(double *angles, double *pos){
    double q1 =  (angles[0] + 1)*HALF_PI;           /* [0,PI] */
    double q2 =  (angles[1] + 1)*HALF_PI;           /* [0,PI] */
    double q3 =  -(angles[2] + 1)*HALF_PI + HALF_PI;/* [PI/2,-PI/2] */
    double q4 =  angles[3]*PI;                      /* [-PI,PI] */
    double q5 = flip*(angles[4] + 1)*HALF_PI;       /* [0,PI] */
    double q6 = angles[5]*PI;                       /* [-PI,PI] */
    double temp = a2 + d1 + d4 + d6;
//    cout << q1/PI << endl;
//    cout << q2/PI << endl;
//    cout << q3/PI << endl;
//    cout << q4/PI << endl;
//    cout << q5/PI << endl;
//    cout << q6/PI << endl;
//    cout <<"---------------"<<endl;
    double sx = cos(q6)*(cos(q4)*sin(q1) - cos(q1)*cos(q2 + q3)*sin(q4)) - (cos(q5)*sin(q1)*sin(q4) + cos(q1)*(cos(q2 + q3)*cos(q4)*cos(q5) - sin(q2 + q3)*sin(q5)))*sin(q6);
    double sy = cos(q1)*(-cos(q4)*cos(q6) + cos(q5)*sin(q4)*sin(q6)) - sin(q1)*(-sin(q2 + q3)*sin(q5)*sin(q6) + cos(q2 + q3)*(cos(q6)*sin(q4) + cos(q4)*sin(q5)*sin(q6)));
    double sz = -cos(q6)*sin(q2 + q3)*sin(q4) - (cos(q4)*cos(q5)*sin(q2 + q3) + cos(q2 + q3)*sin(q5))*sin(q6);
    double ax = sin(q1)*sin(q4)*sin(q5) + cos(q1)*(cos(q5)*sin(q2 + q3) + cos(q2 + q3)*cos(q4)*sin(q5));
    double ay = cos(q5)*sin(q1)*sin(q2 + q3) + (cos(q2 + q3)*cos(q4)*sin(q1) - cos(q1)*sin(q4))*sin(q5);
    double az = -cos(q2 + q3)*cos(q5) + cos(q4)*sin(q2 + q3)*sin(q5);
//    cout << sx << "\t\t" << ax << endl;
//    cout << sy << "\t\t" << ay << endl;
//    cout << sz << "\t\t" << az << endl;
    pos[x_comp] = (1.0/temp)*(d6*sin(q1)*sin(q4)*sin(q5) + cos(q1)*(a2*cos(q2) + (d4 + d6*cos(q5))*sin(q2 + q3) + d6*cos(q2 + q3)*cos(q4)*sin(q5)));
    pos[y_comp] = (1.0/temp)*(cos(q3)*(d4 + d6*cos(q5))*sin(q1)*sin(q2) - d6*(cos(q4)*sin(q1)*sin(q2)*sin(q3) + cos(q1)*sin(q4))*sin(q5) + cos(q2)*sin(q1)*(a2 + (d4 + d6*cos(q5))*sin(q3) + d6*cos(q3)*cos(q4)*sin(q5)));
    pos[z_comp] = (1.0/temp)*(d1 - cos(q2 + q3)*(d4 + d6*cos(q5)) + a2*sin(q2) + d6*cos(q4)*sin(q2 + q3)*sin(q5));
    pos[sx_comp] = sx;
    pos[sy_comp] = sy;
    pos[sz_comp] = sz;
    pos[ax_comp] = ax;
    pos[ay_comp] = ay;
    pos[az_comp] = az;
}


/* input of the NN are the current angles and target pose, output are delta angles to go from current to target angles (just like the Jacobian transpose method) */
void generate::generateDataDelta(int dataPoints, int configs){
    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1, 1);
    /* we want to go from startAngles (initial pose) to target pose*/
    double *deltaAngles; /* angles needed to go from initial pose to target pose */
    double *positions;   /* total input of the NN, target pose + angles of initial pose */
    double *tempAngles;  /* angles of the target pose */
    double *tempPosition;
    double *startAngles; /* angles of the initial pose */

    deltaAngles = (double *)malloc(sizeof(double)*dataPoints*OUTPUT*configs);
    positions = (double *)malloc(sizeof(double)*dataPoints*INPUT*configs);
    /* malloc for arrays with only 15 places.... right why not? */
    tempPosition = (double *)malloc(sizeof(double)*INPUT);
    tempAngles = (double *)malloc(sizeof(double)*OUTPUT);
    startAngles = (double *)malloc(sizeof(double)*OUTPUT);

    double deltas[OUTPUT];

    int i = 0;
    while(i < dataPoints){

        for (int j=0; j<OUTPUT; j++)
            startAngles[j] = dis(gen);
        forwardKinematics(startAngles,tempPosition);
        /* ay_comp > 0 cuz gripper should not point backwards ever*/
        if(tempPosition[y_comp] > 0 && tempPosition[z_comp] > 0 && tempPosition[ay_comp] > 0){

            int k = 0;
            while(k < configs){ /* find configs number of valid target poses */

                for (int j=0; j<OUTPUT; j++) /* generate potential target pose and check it*/
                    tempAngles[j] = dis(gen);
                forwardKinematics(tempAngles,tempPosition);

                if(tempPosition[y_comp] > 0 && tempPosition[z_comp] > 0 && tempPosition[ay_comp] > 0){

                    for(int l = 0; l < OUTPUT; l++)
                        deltas[l] = 0.5*(tempAngles[l] - startAngles[l]); /* scale from -2,2 to -1,1, DO NOT FORGET TO RESCALE LATER! */

                    /* first 9 entries are filled by forwardKinematics, the other 6 inputs of the NN are the initial angles corresponding to the initial pose */
                    memcpy( tempPosition + 9, startAngles, sizeof(double)*6);
                    int shift = i*INPUT*configs + k*INPUT;
                    memcpy(positions + shift ,tempPosition,sizeof(double)*INPUT);
                    /* the output of the network are the deltas by which all the angles have to change */
                    shift = i*OUTPUT*configs + k*OUTPUT;
                    memcpy(deltaAngles + shift, deltas, sizeof(double)*OUTPUT);

                    k++; /* add the valid input/output to the list by incrementing k */
                }
            }
        i++; /* add the valid input/output to the list by incrementing i */
        }
    }
    char data[1024];
    sprintf(data, "pos_delta.dat");
    FILE *file;
    file = fopen(data,"wb");
    for(i=0; i < dataPoints*configs*INPUT; i += INPUT )
        fprintf(file,"%lf\t%lf\t\%lf\t%lf\t%lf\t\%lf\n",positions[i],positions[i+1],positions[i+2],positions[i+6],positions[i+7],positions[i+8]);
    fclose(file);

    struct fann_train_data *train_data = fann_create_train_array(dataPoints*configs, INPUT , positions, OUTPUT, deltaAngles);

    fann_save_train(train_data, "data/ik_test_delta.dat");
    fann_destroy_train(train_data);
    free(deltaAngles); free(positions);
}

void generate::generateDataDeltaFull(int dataPoints, int configs){
    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1, 1);
    /* we want to go from startAngles (initial pose) to delta angles */
    double *deltaAngles; /* angles needed to go from initial pose to target pose */
    double *positions;   /* total input of the NN, error in orientation + angles of initial pose */

    deltaAngles = (double *)malloc(sizeof(double)*dataPoints*OUTPUT*configs);
    positions = (double *)malloc(sizeof(double)*dataPoints*12*configs); /* 6 orientation error entries, 6 initial angles*/

    double startAngles[6];
    double startPos[9]; /* current pos of the arm */
    double desiredPos[9];
    double correctAngles[6]; /* desired angles of the arm */
    double fullError[9];
    double deltas[6];

    int i = 0;
    while(i < dataPoints){

        for (int j=0; j<6; j++)
            startAngles[j] = dis(gen);
        forwardKinematics(startAngles,startPos);
        /* ay_comp > 0 I don't want it to generate a pose with the gripper pointing backwards*/
        if(startPos[y_comp] > 0 && startPos[z_comp] > 0 && startPos[ay_comp] > 0){

            int k = 0;
            while(k < configs){ /* find configs number of valid target poses */

                /* the initial angles should only differ a little from their actual values */
                for (int j=0; j<6; j++){
                    correctAngles[j] = startAngles[j] + dis(gen)/6.0;
                    if(abs(correctAngles[j]) > 1 )
                        correctAngles[j] = startAngles[j] - dis(gen)/6.0;
                }
                forwardKinematics(correctAngles,desiredPos);

                if(desiredPos[y_comp] > 0 && desiredPos[z_comp] > 0 && desiredPos[ay_comp] > 0){
                    /* orientationError will not be in range (-1,1) but it's input so it doesn't matter*/
                    for(int j=0; j<9; j++)
                        fullError[j] = (desiredPos[j] - startPos[j]) ;
                    /* delta will be between -1/4 and 1/4 so we rescale it to (-1,1)*/
                    for(int l = 0; l < 6; l++)
                        deltas[l] = 6*(correctAngles[l] - startAngles[l]);

                    /* first 9 entries are filled by the position+orientation error, the other 6 inputs of the NN are the initial angles*/
                    int shift = i*15*configs + k*15;
                    memcpy(positions + shift ,fullError,sizeof(double)*9);
                    memcpy(positions + shift + 9,startAngles,sizeof(double)*6);

                    /* the output of the network are the deltas by which all the angles have to change */
                    shift = i*6*configs + k*6;
                    memcpy(deltaAngles + shift, deltas, sizeof(double)*6);

                    k++; /* add the valid input/output to the list by incrementing k */
                }
            }
        i++; /* add the valid input/output to the list by incrementing i */
        }
    }


    struct fann_train_data *train_data = fann_create_train_array(dataPoints*configs, 12 , positions, 6, deltaAngles);

    fann_save_train(train_data, "data/ik_train_delta_full.dat");
    fann_destroy_train(train_data);
    free(deltaAngles); free(positions);
}

void generate::generateDataDeltaOrientation(int dataPoints, int configs){
    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1, 1);
    /* we want to go from startAngles (initial pose) to delta angles */
    double *deltaAngles; /* angles needed to go from initial pose to target pose */
    double *positions;   /* total input of the NN, error in orientation + angles of initial pose */

    deltaAngles = (double *)malloc(sizeof(double)*dataPoints*OUTPUT*configs);
    positions = (double *)malloc(sizeof(double)*dataPoints*12*configs); /* 6 orientation error entries, 6 initial angles*/

    double startAngles[6];
    double startPos[9]; /* current pos of the arm */
    double desiredPos[9];
    double correctAngles[6]; /* desired angles of the arm */
    double orientationError[6];
    double deltas[6];

    int i = 0;
    while(i < dataPoints){

        for (int j=0; j<6; j++)
            startAngles[j] = dis(gen);
        forwardKinematics(startAngles,startPos);
        /* ay_comp > 0 I don't want it to generate a pose with the gripper pointing backwards*/
        if(startPos[y_comp] > 0 && startPos[z_comp] > 0 && startPos[ay_comp] > 0){

            int k = 0;
            while(k < configs){ /* find configs number of valid target poses */

                /* the initial angles should only differ a little from their actual values */
                for (int j=0; j<6; j++){
                    correctAngles[j] = startAngles[j] + dis(gen)/4.0;
                    if(abs(correctAngles[j]) > 1 )
                        correctAngles[j] = startAngles[j] - dis(gen)/4.0;
                }
                forwardKinematics(correctAngles,desiredPos);

                if(desiredPos[y_comp] > 0 && desiredPos[z_comp] > 0 && desiredPos[ay_comp] > 0){
                    /* orientationError will not be in range (-1,1) but it's input so it doesn't matter*/
                    for(int j=0; j<6; j++)
                        orientationError[j] = (desiredPos[j+3] - startPos[j+3]) ;
                    /* delta will be between -1/4 and 1/4 so we rescale it to (-1,1)*/
                    for(int l = 0; l < 6; l++)
                        deltas[l] = 4*(correctAngles[l] - startAngles[l]);

                    /* first 6 entries are filled by the orientation error, the other 6 inputs of the NN are the initial angles*/
                    int shift = i*12*configs + k*12;
                    memcpy(positions + shift ,orientationError,sizeof(double)*6);
                    memcpy(positions + shift + 6,startAngles,sizeof(double)*6);

                    /* the output of the network are the deltas by which all the angles have to change */
                    shift = i*6*configs + k*6;
                    memcpy(deltaAngles + shift, deltas, sizeof(double)*6);

                    k++; /* add the valid input/output to the list by incrementing k */
                }
            }
        i++; /* add the valid input/output to the list by incrementing i */
        }
    }

    struct fann_train_data *train_data = fann_create_train_array(dataPoints*configs, 12 , positions, 6, deltaAngles);

    fann_save_train(train_data, "data/ik_test_delta_orientation.dat");
    fann_destroy_train(train_data);
    free(deltaAngles); free(positions);
}

void generate::generateDataDeltaPosition(int dataPoints, int configs){
    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1, 1);
    /* we want to go from startAngles (initial pose) to target pose*/
    double *deltaAngles; /* angles needed to go from initial pose to target pose */
    double *positions;   /* total input of the NN, error in orientation + angles of initial pose */

    deltaAngles = (double *)malloc(sizeof(double)*dataPoints*OUTPUT*configs);
    positions = (double *)malloc(sizeof(double)*dataPoints*9*configs); /* 3 position error entries, 6 initial angles*/

    double startAngles[6];
    double startPos[9]; /* current pos of the arm */
    double desiredPos[9];
    double correctAngles[6]; /* desired angles of the arm */
    double positionError[3];
    double deltas[6];

    int i = 0;
    while(i < dataPoints){

        for (int j=0; j<6; j++)
            startAngles[j] = dis(gen);
        forwardKinematics(startAngles,startPos);
        /* ay_comp > 0 cuz gripper should not point backwards ever*/
        if(startPos[y_comp] > 0 && startPos[z_comp] > 0 && startPos[ay_comp] > 0){

            int k = 0;
            while(k < configs){ /* find configs number of valid target poses */

                /* the initial angles should only differ a little from their actual values */
                for (int j=0; j<6; j++){
                    correctAngles[j] = startAngles[j] + dis(gen)/4.0;
                    if(abs(correctAngles[j]) > 1 )
                        correctAngles[j] = startAngles[j] - dis(gen)/4.0;
                }
                forwardKinematics(correctAngles,desiredPos);

                if(desiredPos[y_comp] > 0 && desiredPos[z_comp] > 0 && desiredPos[ay_comp] > 0){
                    /* since the angles change only a little this error shouldn't need a correction*/
                    for(int j=0; j<3; j++)
                        positionError[j] = (desiredPos[j] - startPos[j]); /* biggest positionError is only 0.5, rescale needed? */
                    /* delta should be between -1/4 and 1/4 so we rescale it*/
                    for(int l = 0; l < 6; l++)
                        deltas[l] = 4*(correctAngles[l] - startAngles[l]);

                    /* first 3 entries are filled by the position error, the other 6 inputs of the NN are the initial angles*/
                    int shift = i*9*configs + k*9;
                    memcpy(positions + shift ,positionError,sizeof(double)*3);
                    memcpy(positions + shift + 3,startAngles,sizeof(double)*6);

                    /* the output of the network are the deltas by which all the angles have to change */
                    shift = i*6*configs + k*6;
                    memcpy(deltaAngles + shift, deltas, sizeof(double)*6);

                    k++; /* add the valid input/output to the list by incrementing k */
                }
            }
        i++; /* add the valid input/output to the list by incrementing i */
        }
    }

    struct fann_train_data *train_data = fann_create_train_array(dataPoints*configs, 9 , positions, 6, deltaAngles);

    fann_save_train(train_data, "data/ik_test_delta_position.dat");
    fann_destroy_train(train_data);
    free(deltaAngles); free(positions);
}

void generate::generateData(int dataPoints){
    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1, 1);

    double angles[dataPoints*OUTPUT];
    double positions[dataPoints*INPUT];
    double tempAngles[OUTPUT];
    double tempPositions[INPUT];

    int i = 0;
    int t = 0;
    while(i < dataPoints*OUTPUT){

        for (int j=0; j<OUTPUT; j++)
            angles[i+j] = dis(gen);

        memcpy(tempAngles,&angles[i],sizeof(double) * OUTPUT);

        forwardKinematics(tempAngles,tempPositions);

        memcpy(positions + t ,tempPositions,sizeof(double)*(OUTPUT+3));

        if(positions[t + z_comp]>0){
            i+=OUTPUT;   /* number of angles*/
            t+=OUTPUT+3; /* number of "positions" */
        }
    }
    char data[1024];
    sprintf(data, "pos.dat");
    FILE *file;
    file = fopen(data,"wb");
    for(i=0; i<dataPoints*OUTPUT; i+=(OUTPUT+3)){
        fprintf(file,"%lf\t%lf\t\%lf\t%lf\t%lf\t\%lf\n",positions[i],positions[i+1],positions[i+2],positions[i+6],positions[i+7],positions[i+8]);
    }
    fclose(file);

    struct fann_train_data *train_data = fann_create_train_array(dataPoints, OUTPUT+3 , positions, OUTPUT, angles);

    fann_save_train(train_data, "data/ik_train.dat");
    fann_destroy_train(train_data);
}
