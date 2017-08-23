#include <stdio.h>
#include <random>
#include <iostream>
#include <math.h>
#include <chrono>
#include "doublefann.h"
#include <string.h>

#define x_comp      0
#define y_comp      1
#define z_comp      2
#define alpha_comp  3
#define beta_comp   4
#define gamma_comp  5

#define flip 1

#define PI (3.141592653589793)
#define HALF_PI (1.570796326794897)
#define DOF 6 /* degrees of freedom in the system */

using namespace::std;

#define d1  12.5   //ground to q1
#define d6  12.0 //gripper to wrist
#define a2 15.0    //q1 to q2
#define d4 19.2  //q2 to wrist

/* takes in array with range [-1,1] and outputs array with range [-1,1].*/
void forwardKinematics(double *angles, double *pos){

    double q1 =  (angles[0] + 1)*HALF_PI;           /* [0,2PI] */
    double q2 =  (angles[1] + 1)*HALF_PI;           /* [0,2PI] */
    double q3 =  -(angles[2] + 1)*HALF_PI + HALF_PI;/* [PI/2,-PI/2] */
    double q4 =  angles[3]*PI;                      /* [-PI,PI] */
    double q5 = flip*(angles[4] + 1)*HALF_PI;       /* [0,PI] */
    double q6 = angles[5]*PI;                       /* [-PI,PI] */
    double temp = 1;//a2 + d1 + d4 + d6;
    cout << q1/PI << endl;
    cout << q2/PI << endl;
    cout << q3/PI << endl;
    cout << q4/PI << endl;
    cout << q5/PI << endl;
    cout << q6/PI << endl;
    cout <<"---------------"<<endl;

    /* the elements of the rotation matrix gathered from the permutated one, see documentation. */
    /* for instance r21 is actually r22*/
    double r21 = cos(q1)*(-cos(q4)*cos(q6) + cos(q5)*sin(q4)*sin(q6)) - sin(q1)*(-sin(q2 + q3)*sin(q5)*sin(q6) + cos(q2 + q3)*(cos(q6)*sin(q4) + cos(q4)*sin(q5)*sin(q6)));
    double r11 = cos(q6)*(cos(q4)*sin(q1) - cos(q1)*cos(q2 + q3)*sin(q4)) - (cos(q5)*sin(q1)*sin(q4) + cos(q1)*(cos(q2 + q3)*cos(q4)*cos(q5) - sin(q2 + q3)*sin(q5)))*sin(q6);
    double r31 = -cos(q6)*sin(q2 + q3)*sin(q4) - (cos(q4)*cos(q5)*sin(q2 + q3) + cos(q2 + q3)*sin(q5))*sin(q6);
    double r32 = -cos(q2 + q3)*cos(q5) + cos(q4)*sin(q2 + q3)*sin(q5);
    double r33 = cos(q4)*cos(q5)*cos(q6)*sin(q2 + q3) + cos(q2 + q3)*cos(q6)*sin(q5) - sin(q2 + q3)*sin(q4)*sin(q6);

    pos[x_comp] = (1.0/temp)*(d6*sin(q1)*sin(q4)*sin(q5) + cos(q1)*(a2*cos(q2) + (d4 + d6*cos(q5))*sin(q2 + q3) + d6*cos(q2 + q3)*cos(q4)*sin(q5)));
    pos[y_comp] = (1.0/temp)*(cos(q3)*(d4 + d6*cos(q5))*sin(q1)*sin(q2) - d6*(cos(q4)*sin(q1)*sin(q2)*sin(q3) + cos(q1)*sin(q4))*sin(q5) + cos(q2)*sin(q1)*(a2 + (d4 + d6*cos(q5))*sin(q3) + d6*cos(q3)*cos(q4)*sin(q5)));
    pos[z_comp] = (1.0/temp)*(d1 - cos(q2 + q3)*(d4 + d6*cos(q5)) + a2*sin(q2) + d6*cos(q4)*sin(q2 + q3)*sin(q5));
    pos[alpha_comp] = (1.0/PI)*atan2(r21,r11);
    pos[beta_comp]  = (1.0/PI)*atan2(-r31,sqrt(pow(r32,2) + pow(r33,2)));
    pos[gamma_comp] = (1.0/PI)*atan2(r32,r33);
}

void generateData(int dataPoints){
    std::random_device rd;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(-1, 1);

    fann_type angles[dataPoints*DOF];
    fann_type positions[dataPoints*DOF];
    fann_type tempAngles[DOF];
    fann_type tempPositions[DOF];

    int i =0;
    while(i < dataPoints*DOF){
        for (int j=0; j<DOF; j++)
            angles[i+j] = dis(gen);
        memcpy(tempAngles,&angles[i],sizeof(double) * DOF);

        forwardKinematics(tempAngles,tempPositions);

        memcpy(positions + i,tempPositions,sizeof(double) * DOF);
        if(positions[i+2]>0)
            i+=DOF;
    }
    char data[1024];
    sprintf(data, "pos.dat");
    FILE *file;
    file = fopen(data,"wb");
    for(i=0; i<dataPoints*DOF; i+=DOF){
        fprintf(file,"%lf\t%lf\t\%lf\n",positions[i],positions[i+1],positions[i+2]);
    }
    fclose(file);

    struct fann_train_data *train_data = fann_create_train_array(dataPoints, DOF , positions, DOF, angles);

    fann_save_train(train_data, "ik_test.dat");
    fann_destroy_train(train_data);
}

void trainNetwork(unsigned int num_layers, unsigned int *topology){

	const double desired_error = (const double) 0.0001;
	const unsigned int max_epochs = 100000;
	const unsigned int epochs_between_res = 2500;
	int max_runs = max_epochs/epochs_between_res;
	struct fann *ann = fann_create_standard_array(num_layers, topology);
    struct fann_train_data *train_data, *test_data;

    train_data = fann_read_train_from_file("ik_train.dat");
    test_data = fann_read_train_from_file("ik_test.dat");
    /* network settings */
    fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);
	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_LINEAR);
    fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.1f);
	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
    //fann_init_weights(ann, train_data);

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
	/* train the network and save intermediate results */
    for(int j = 0; j < max_runs; j++){
        fann_train_on_data(ann, train_data, epochs_between_res, 0, desired_error);

        fann_reset_MSE(ann);
        for(unsigned int i = 0; i < fann_length_train_data(test_data); i++)
            fann_test(ann, test_data->input[i], test_data->output[i]);
        MSE = fann_get_MSE(ann);
        cout << "epochs= " << (j+1)*epochs_between_res << "\t MSE= " << MSE << endl;
        cout << 100*(j+1)/(float)max_runs << "%" << endl;
        //printf("MSE error on test data: %f\n", MSE);
        fprintf(outFile, "%d\t%lf \n", (j+1)*epochs_between_res, MSE);
    }
	fann_save(ann, name);
    /* cleanup */
	fann_destroy(ann);
	fann_destroy_train(train_data);
	fann_destroy_train(test_data);
	fclose(outFile);
}

int main(){
//    generateData(500);
//
//    unsigned int layers[3] = {DOF,200,DOF};
//    trainNetwork(3, layers);

//    unsigned int layers2[4] = {2,50,50,2};
//    trainNetwork(4, layers2);

//	testNetwork("ik_float_50.net");

//    struct fann *ann = fann_create_from_file("ik_float_15.net");
//    cout << checkNN(ann) << endl;

    double pos[6];
    double angles[6] = {0,0,0,0.5,-1,0};
    forwardKinematics(angles,pos);
    for(int i=0; i<6; i++)
        cout << pos[i] << endl;

	return 0;
}
