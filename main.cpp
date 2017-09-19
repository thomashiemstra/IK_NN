#include <stdio.h>
#include <random>
#include <iostream>
#include <math.h>
#include <chrono>
#include "doublefann.h"
#include <string.h>
#include <chrono>
#include "generate.h"


using namespace::std;

generate gen = generate();

void trainNetwork(unsigned int num_layers, unsigned int *topology){
	const double desired_error = (const double) 0.0001;
	const unsigned int max_epochs = 10000;
	const unsigned int epochs_between_res = 100;
	int max_runs = max_epochs/epochs_between_res;

	//struct fann *ann = fann_create_standard_array(num_layers, topology);
	//struct fann *ann = fann_create_sparse_array(0.5, num_layers, topology);
	struct fann *ann = fann_create_shortcut_array(num_layers, topology);

	//struct fann *ann = fann_create_from_file("nn/ik_float_orientation_40_40.net");

    struct fann_train_data *train_data, *test_data;
    train_data = fann_read_train_from_file("data/ik_train_delta_position.dat");
    test_data = fann_read_train_from_file("data/ik_test_delta_position.dat");
    /* network settings */
    fann_set_activation_steepness_hidden(ann, 0.5);
	fann_set_activation_steepness_output(ann, 0.5);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.1f);

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
    //fann_set_training_algorithm(ann, FANN_TRAIN_BATCH);

    //fann_set_train_error_function(ann,FANN_ERRORFUNC_TANH);

    FILE *outFile;
    char name[1024];
	char temp[1024];
	char tempName[1024];
	sprintf(name,"nn/ik_float_position");
	sprintf(tempName, "res/res_position");
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

void setWeightsTest(){
    unsigned int layers[3] = {1,2,1};
    unsigned int num_connections;
    struct fann_connection *connections;
    struct fann_connection *new_connections;

    struct fann *ann = fann_create_standard_array(3, layers);
    fann_randomize_weights(ann,-1,1);

    num_connections = fann_get_total_connections(ann);
    connections     = (struct fann_connection *)malloc(sizeof(struct fann_connection) * num_connections);
    new_connections = (struct fann_connection *)malloc(sizeof(struct fann_connection) * num_connections);

    fann_get_connection_array(ann, connections);
    new_connections = connections;

    for(unsigned int i=0; i<num_connections; i++){
        cout << connections[i].weight << endl;
        new_connections[i].weight = 2;
    }

    fann_set_weight_array(ann,new_connections,num_connections);

    fann_get_connection_array(ann, connections);

    for(unsigned int i=0; i<num_connections; i++)
        cout << connections[i].weight << endl;


    free(connections);
    free(new_connections);
}

int main(){



	return 0;
}
