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

using namespace::std;

#define d1  12.5   //ground to q1
#define d6  12.0   //gripper to wrist
#define a2 15.0    //q1 to q2
#define d4 19.2    //q2 to wrist

/* takes in array with range [-1,1] and outputs array with range [-1,1]. Orientation is given by the approach and sliding direction of the gripper. technically 3DOF but 6 outputs */
void forwardKinematics(double *angles, double *pos){
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
void generateDataDelta(int dataPoints, int configs){
    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1, 1);
    /* we want to go from startAngles (initial pose) to target pose*/
    fann_type *deltaAngles; /* angles needed to go from initial pose to target pose */
    fann_type *positions;   /* total input of the NN, target pose + angles of initial pose */
    fann_type *tempAngles;  /* angles of the target pose */
    fann_type *tempPosition;
    fann_type *startAngles; /* angles of the initial pose */

    deltaAngles = (fann_type *)malloc(sizeof(fann_type)*dataPoints*OUTPUT*configs);
    positions = (fann_type *)malloc(sizeof(fann_type)*dataPoints*INPUT*configs);
    /* malloc for arrays with only 15 places.... right why not? */
    tempPosition = (fann_type *)malloc(sizeof(double)*INPUT);
    tempAngles = (fann_type *)malloc(sizeof(fann_type)*OUTPUT);
    startAngles = (fann_type *)malloc(sizeof(fann_type)*OUTPUT);

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
                    memcpy( tempPosition + 9, startAngles, sizeof(fann_type)*6);
                    int shift = i*INPUT*configs + k*INPUT;
                    memcpy(positions + shift ,tempPosition,sizeof(fann_type)*INPUT);
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

    fann_save_train(train_data, "ik_test_delta.dat");
    fann_destroy_train(train_data);
    free(deltaAngles); free(positions);
}

void generateData(int dataPoints){
    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1, 1);

    fann_type angles[dataPoints*OUTPUT];
    fann_type positions[dataPoints*INPUT];
    fann_type tempAngles[OUTPUT];
    fann_type tempPositions[INPUT];

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

    fann_save_train(train_data, "ik_train.dat");
    fann_destroy_train(train_data);
}

void trainNetwork(unsigned int num_layers, unsigned int *topology){
	const double desired_error = (const double) 0.0001;
	const unsigned int max_epochs = 10000;
	const unsigned int epochs_between_res = 100;
	int max_runs = max_epochs/epochs_between_res;

	//struct fann *ann = fann_create_standard_array(num_layers, topology);
	//struct fann *ann = fann_create_sparse_array(0.2, num_layers, topology);
	//struct fann *ann = fann_create_shortcut_array(num_layers, topology);
	struct fann *ann = fann_create_from_file("nn/ik_float_40_20.net");

    struct fann_train_data *train_data, *test_data;
    train_data = fann_read_train_from_file("ik_train_delta.dat");
    test_data = fann_read_train_from_file("ik_test_delta.dat");
    /* network settings */
    fann_set_activation_steepness_hidden(ann, 0.2);
	fann_set_activation_steepness_output(ann, 0.2);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.1f);
	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

    fann_set_train_error_function(ann,FANN_ERRORFUNC_TANH);

    FILE *outFile;
    char name[1024];
	char temp[1024];
	char tempName[1024];
	sprintf(name,"nn/ik_float");
	sprintf(tempName, "res/res");
	for(unsigned int i=0; i<num_layers-2; i++){
        sprintf(temp,"_%d",topology[i+1]);
        strcat(name, temp);
        strcat(tempName, temp);
	}
	strcat(name, ".net");
	strcat(tempName, ".dat");
	cout << name << endl;

	outFile = fopen(tempName,"wb");
    double MSE =0;

    auto begin = std::chrono::high_resolution_clock::now();
	/* train the network and save intermediate results */
    for(int j = 0; j < max_runs; j++){
        fann_train_on_data(ann, train_data, epochs_between_res, 0, desired_error);

        fann_reset_MSE(ann);
        for(unsigned int i = 0; i < fann_length_train_data(test_data); i++)
            fann_test(ann, test_data->input[i], test_data->output[i]);
        MSE = fann_get_MSE(ann);

        auto end = std::chrono::high_resolution_clock::now();
        cout << "epochs= " << (j+1)*epochs_between_res << "\t MSE= " << MSE << endl;
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();
        float percentage = 100*(j+1)/(float)max_runs;
        cout << percentage << "% time left:" << (1.0/60000)*((time*(100/percentage)) - time) << " mins" << endl;
        //printf("MSE error on test data: %f\n", MSE);
        fprintf(outFile, "%d\t%lf \n", (j+1)*epochs_between_res, MSE);
        fann_save(ann, name);
    }
    /* cleanup */
	fann_destroy(ann);
	fann_destroy_train(train_data);
	fann_destroy_train(test_data);
	fclose(outFile);
}

int main(){
//   generateDataDelta(200,200);

    unsigned int layers[4] = {INPUT,40,20,OUTPUT};
    trainNetwork(4, layers);

//    unsigned int layers1[5] = {INPUT,20,20,10,OUTPUT};
//    trainNetwork(5, layers1);


//    double ps[9];
//    double angles[6] = {-0.151387, -0.324997, 0.723476, 0.186593, -0.255015, 0.098493};
//    forwardKinematics(angles,pos);
//    for(int i=0; i<9; i++){
//        cout << pos[i] << "  ";
//    }

	return 0;
}
