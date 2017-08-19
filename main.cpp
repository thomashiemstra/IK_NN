#include <stdio.h>
#include <random>
#include <iostream>
#include <math.h>
#include <chrono>
#include "doublefann.h"
#include <string.h>

#define PI (3.141592653589793)
#define HALF_PI (1.570796326794897)

using namespace::std;

double l1 = 10.0;
double l2 = 10.0;

/* takes in array with range [-1,1] and outputs array with range [-1,1].*/
void forwardKinematics(double q1, double q2, double& pos1, double& pos2){

    q1 =  (q1 + 1)*HALF_PI;
    q2 =  -(q2 + 1)*HALF_PI;

    pos1 = (l1*cos(q1) + l2*cos(q1 + q2)); /* x */
    pos2 = (l1*sin(q1) + l2*sin(q1 + q2)); /* y */

    double temp = l1 + l2;
    pos1 = (pos1)/temp;
    pos2 = (pos2)/temp;
    //cout << "y= " << pos2 << endl;
}
/* -1 for elbow up, 1 for down */
void generateData(int dataPoints, int elbow){

    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-1, 1);

    fann_type angles[dataPoints*2];
    fann_type positions[dataPoints*2];

    int i =0;
    while(i < dataPoints*2){
        angles[i] = dis(gen);
        angles[i+1] = dis(gen);
        forwardKinematics(angles[i],angles[i+1],positions[i],positions[i+1]);

        if(positions[i+1]>0)
            i+=2;
    }

    char data[1024];
    sprintf(data, "pos.dat");
    FILE *file;
    file = fopen(data,"wb");
    for(i=0; i<dataPoints*2; i+=2){
        fprintf(file,"%lf \t %lf\n",positions[i],positions[i+1]);
    }
    fclose(file);

    struct fann_train_data *train_data = fann_create_train_array(dataPoints, 2 , positions, 2, angles);

    fann_save_train(train_data, "ik_train.dat");
    fann_destroy_train(train_data);
}

void testNetwork(char* name){
	unsigned int i;

	struct fann *ann;
	struct fann_train_data *test_data;

	printf("Creating network.\n");

	ann = fann_create_from_file(name);

	//fann_print_connections(ann);
	//fann_print_parameters(ann);

	printf("Testing network.\n");

	test_data = fann_read_train_from_file("ik_test.dat");

	//cout << test_data->input[i] << endl;

	fann_reset_MSE(ann);
	for(i = 0; i < fann_length_train_data(test_data); i++)
	{
		fann_test(ann, test_data->input[i], test_data->output[i]);
	}
    printf("MSE error on test data: %f\n", fann_get_MSE(ann));

	fann_destroy_train(test_data);
	fann_destroy(ann);

}

void trainNetwork(unsigned int num_layers, unsigned int *topology){

	const double desired_error = (const double) 0.0001;
	const unsigned int max_epochs = 20000;
	const unsigned int epochs_between_reports = 1000;


	struct fann *ann = fann_create_standard_array(num_layers, topology);
    struct fann_train_data *data;

    data = fann_read_train_from_file("ik_train.dat");

    fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_LINEAR);

    fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.1f);

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

    fann_init_weights(ann, data);
	fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);
    /* put the topology of the network in the file name */
    char name[1024];
	char temp[1024];
	sprintf(name,"ik_float");
	for(int i=0; i<num_layers-2; i++){
        sprintf(temp,"_%d",topology[i+1]);
        strcat(name, temp);
	}
	strcat(name, ".net");

	fann_save(ann, name);

	fann_destroy(ann);
	fann_destroy_train(data);
}

void trainNetworkCascade(){
	const unsigned int num_input = 2;
	const unsigned int num_output = 2;
	const double desired_error = (const double) 0.0001;
	unsigned int max_neurons = 100;
	unsigned int neurons_between_reports = 1;
	const unsigned int max_epochs = 10000;

	struct fann *ann = fann_create_shortcut(2, num_input,  num_output);
    struct fann_train_data *data;
    fann_type steepness;
    enum fann_activationfunc_enum activation;

    data = fann_read_train_from_file("ik_train.dat");

    fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_LINEAR);

    fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.1f);

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
	fann_set_train_error_function(ann, FANN_ERRORFUNC_LINEAR);

    fann_init_weights(ann, data);

    steepness = 1;
    fann_set_cascade_activation_steepnesses(ann, &steepness, 1);

    activation = FANN_SIGMOID_SYMMETRIC;

    fann_set_cascade_activation_functions(ann, &activation, 1);
    fann_set_cascade_num_candidate_groups(ann, 8);

    fann_cascadetrain_on_data(ann, data, max_neurons, neurons_between_reports, desired_error);

    fann_save(ann, "cascade_train.net");

    printf("Cleaning up.\n");
	fann_destroy_train(data);
	fann_destroy(ann);
}

void ikNN(double x, double y, double angles[2]){

    fann_type *calc_out;
    fann_type input[2];

    struct fann *ann = fann_create_from_file("ik_float_50.net");

    input[0] = x;
    input[1] = y;

    calc_out = fann_run(ann,input);

    printf("IK test (%lf,%lf) -> %lf %lf\n", input[0], input[1], calc_out[0], calc_out[1]);

    double pos[2];

    forwardKinematics(calc_out[0], calc_out[1],pos[0],pos[1]);
    cout << pos[0] << "  " << pos[1] << endl;
    cout << "MSE= " <<sqrt( pow((pos[0]-input[0]),2) + pow((pos[1]-input[1]),2) ) << endl;

    angles[0] = calc_out[0];
    angles[1] = calc_out[1];

}

int main(){


//    generateData(1000,-1);

//    unsigned int layers[] = {2,50,2};
//    unsigned int num_layers = sizeof(layers)/sizeof(layers[0]);
//    trainNetwork(num_layers, layers);

//	testNetwork("ik_float_50.net");

//    double angles[2];
//    ikNN(0.5,0.5,angles);

	return 0;
}
