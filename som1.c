// Description:	Self-Organizing Map Network Example 1.
// http://mnemstudio.org/ai/nn/som_cpp_ex1.txt
// http://mnemstudio.org/neural-networks-som1.htm

// $ gcc som1.c -o som1.exe -lm
// $ ./som1.exe


// A Kohonen Self-Organizing Network with 4 Inputs and 2-Node Linear Array of Cluster Units(output nodes).
// The results will vary slightly with different combinations of learning rate, decay rate, and alpha value.

/*
Vector: 4 vertical elements in an array
Takes 4 training patterns {1, 1, 0, 0}, {0, 0, 0, 1}, {1, 0, 0, 0}, {0, 0, 1, 1}
Organizes (groups/clusters) the training patterns into one of two outputs by adjusting the weights and the 4 training inputs {1, 1, 0, 0}, {0, 0, 0, 1}, {1, 0, 0, 0}, {0, 0, 1, 1}
Tests the organization by classifying/grouping/clustering 4 previously unseen inputs and outputing the correct group each test input belongs (1 or 2) {1, 0, 0, 1}, {0, 1, 1, 0}, {1, 0, 1, 0}, {0, 1, 0, 1}


	Name:		
	Purpose:		
	Arguments:	
	Called By: 	
	Calls To: 	
				
	Notes:		
				
				
	Returns: 		
	Pseudocode:	
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				
				

*/

#include <stdio.h>
#include <math.h>

/////////////////////////// Declares //////////////////////////////////////////////////////
#define outputs		2			// 2 outputs
#define trainvect		4			// 4 training vectors
#define testvect		4			// 4 testing vectors
#define vectlen		4			// 4 vertical elements is a vector
#define decayRate		0.96			
#define minAlpha		0.01			
#define alpha			0.06

/////////////////////////// Globals ///////////////////////////////////////////////////////

double gwt[vectlen][outputs] = {{0.2, 0.6}, {0.5, 0.9}, {0.8, 0.4}, {0.7, 0.3}} ;				// A 4(long) x 2(wide) array of weights (2 vectors)

int gtrainpat[vectlen][trainvect] = {{1, 1, 0, 0}, {0, 0, 0, 1}, {1, 0, 0, 0}, {0, 0, 1, 1}} ;		// A 4(long) x 4 (wide) array of training patterns (4 vectors)

int gtestpat[vectlen][testvect] = {{1, 0, 0, 1}, {0, 1, 1, 0}, {1, 0, 1, 0}, {0, 1, 0, 1}} ;		// A 4(long) x 4 (wide) array of inputs (4 vectors)

/////////////////////////// Prototypes ////////////////////////////////////////////////////
void training() ;
void testing() ;



int main()
{


	training() ;
	printf("alpha now = %lf\n", alpha) ;
	
	testing() ;
	
	return 0 ;
}


void training()
{
/*
	Name:		training
	Purpose:		Train the SOM
	Arguments:	None
	Called By: 	main
	Calls To: 	computeInput
				minimum
	Notes:		A vector is 4 vertical elements of the 
				alpha is a global variable that defines the learning rate
				
	Returns: 		None
	Pseudocode:	While alpha > 0
					increment itterations
					
					for each element in the current vector
						Call computeInput
						See which is smaller, d[0] or d[1]?
						Update the weights on the winning unit.
						For each weight in the vector
							 w[dMin][i] = w[dMin][i] + (alpha * (pattern[vecNum][i] - w[dMin][i])) ;
				
					Reduce the learning rate.
				
*/

int i = 0 ;
int j = 0 ;
int iterations = 0 ;
int vectnum = 0 ;
double tmpalpha = alpha ;
double dist1 = 0.0 ;
double dist2 = 0.0 ;

	printf("\n") ;
	for (i=0; i<vectlen; i++)
		printf("%lf	%lf\n", gwt[i][0], gwt[i][1]) ;
	

	while(tmpalpha > minAlpha)
	{
		for(j=0; j<trainvect; j++)	// For the 2 outputs
		{
			dist1 = 0 ;
			dist2 = 0 ;
			
			for(i=0; i<trainvect; i++)			// For the 4 training inputs (0, 1, 2, 3)
			{
				dist1 += pow((gwt[i][0] - gtrainpat[i][j]), 2) ;	// d[i] += (weight[i][j] - tests[current training vector component][j])^2
				dist2 += pow((gwt[i][1] - gtrainpat[i][j]), 2) ;	// d[i] += (weight[i][j] - tests[current training vector component][j])^2
			}
			
			if (dist1 < dist2)		// Update gwt[0][x] weight values
				for (i=0; i<trainvect; i++)
					gwt[i][0] = gwt[i][0] + (tmpalpha * (gtrainpat[i][j] - gwt[i][0])) ;
			else
				for (i=0; i<trainvect; i++)
					gwt[i][1] = gwt[i][1] + (tmpalpha * (gtrainpat[i][j] - gwt[i][1])) ;
		}
		
		tmpalpha *= decayRate ;	// Decrease the learning rate
		printf("tmpAlpha = %lf	dist1=%lf	dist2=%lf\n", tmpalpha, dist1, dist2) ;
		i++ ;
	}
	printf("\n") ;
	for (i=0; i<vectlen; i++)
		printf("%lf	%lf\n", gwt[i][0], gwt[i][1]) ;
	
	
}


void testing()
{
int i = 0 ;
int j = 0 ;
int vectnum = 0 ;
double dist1 = 0 ;
double dist2 = 0 ;

	printf("\nTesting\n") ;
	
	for(j=0; j<testvect; j++)			// For the 4 testing inputs (0, 1, 2, 3)
	{
		dist1 = 0 ;
		dist2 = 0 ;
		
		for(i=0; i<vectlen; i++)			// For the 4 testing inputs (0, 1, 2, 3)
		{
			dist1 += pow((gwt[i][0] - gtestpat[j][i]), 2) ;	// d[i] += (weight[i][j] - tests[current training vector component][j])^2
			dist2 += pow((gwt[i][1] - gtestpat[j][i]), 2) ;	// d[i] += (weight[i][j] - tests[current training vector component][j])^2
		}
		
		if (dist1 < dist2)
			printf("%d,%d,%d,%d BMU category 1		distance1 %lf		distance2 %lf\n", gtestpat[j][0], gtestpat[j][1], gtestpat[j][2], gtestpat[j][3], dist1, dist2) ;
		else
			printf("%d,%d,%d,%d BMU category 2		distance1 %lf		distance2 %lf\n", gtestpat[j][0], gtestpat[j][1], gtestpat[j][2], gtestpat[j][3], dist1, dist2) ;
	}
}

/*
[localhost som1]$ gcc som1.c -o som1.exe -lm
[localhost som1]$ ./som1.exe

0.200000	0.600000
0.500000	0.900000
0.800000	0.400000
0.700000	0.300000
tmpAlpha = 0.057600	dist1=0.397272	dist2=2.101478
tmpAlpha = 0.055296	dist1=0.340382	dist2=2.183191
tmpAlpha = 0.053084	dist1=0.300151	dist2=2.262043
tmpAlpha = 0.050961	dist1=0.271705	dist2=2.336365
tmpAlpha = 0.048922	dist1=0.251644	dist2=2.405371
tmpAlpha = 0.046965	dist1=0.237575	dist2=2.468813
tmpAlpha = 0.045087	dist1=0.227807	dist2=2.526767
tmpAlpha = 0.043283	dist1=0.221134	dist2=2.579486
tmpAlpha = 0.041552	dist1=0.216689	dist2=2.627323
tmpAlpha = 0.039890	dist1=0.213851	dist2=2.670669
tmpAlpha = 0.038294	dist1=0.212169	dist2=2.709921
tmpAlpha = 0.036763	dist1=0.211314	dist2=2.745468
tmpAlpha = 0.035292	dist1=0.211048	dist2=2.777676
tmpAlpha = 0.033880	dist1=0.211196	dist2=2.806880
tmpAlpha = 0.032525	dist1=0.211632	dist2=2.833389
tmpAlpha = 0.031224	dist1=0.212261	dist2=2.857481
tmpAlpha = 0.029975	dist1=0.213015	dist2=2.879406
tmpAlpha = 0.028776	dist1=0.213846	dist2=2.899388
tmpAlpha = 0.027625	dist1=0.214717	dist2=2.917626
tmpAlpha = 0.026520	dist1=0.215601	dist2=2.934297
tmpAlpha = 0.025459	dist1=0.216482	dist2=2.949561
tmpAlpha = 0.024441	dist1=0.217345	dist2=2.963557
tmpAlpha = 0.023463	dist1=0.218183	dist2=2.976412
tmpAlpha = 0.022525	dist1=0.218989	dist2=2.988235
tmpAlpha = 0.021624	dist1=0.219760	dist2=2.999126
tmpAlpha = 0.020759	dist1=0.220494	dist2=3.009173
tmpAlpha = 0.019928	dist1=0.221190	dist2=3.018456
tmpAlpha = 0.019131	dist1=0.221848	dist2=3.027043
tmpAlpha = 0.018366	dist1=0.222469	dist2=3.035000
tmpAlpha = 0.017631	dist1=0.223055	dist2=3.042380
tmpAlpha = 0.016926	dist1=0.223606	dist2=3.049236
tmpAlpha = 0.016249	dist1=0.224124	dist2=3.055612
tmpAlpha = 0.015599	dist1=0.224611	dist2=3.061549
tmpAlpha = 0.014975	dist1=0.225069	dist2=3.067084
tmpAlpha = 0.014376	dist1=0.225498	dist2=3.072250
tmpAlpha = 0.013801	dist1=0.225901	dist2=3.077077
tmpAlpha = 0.013249	dist1=0.226280	dist2=3.081592
tmpAlpha = 0.012719	dist1=0.226635	dist2=3.085819
tmpAlpha = 0.012210	dist1=0.226969	dist2=3.089780
tmpAlpha = 0.011722	dist1=0.227283	dist2=3.093496
tmpAlpha = 0.011253	dist1=0.227577	dist2=3.096986
tmpAlpha = 0.010803	dist1=0.227854	dist2=3.100265
tmpAlpha = 0.010371	dist1=0.228114	dist2=3.103349
tmpAlpha = 0.009956	dist1=0.228359	dist2=3.106252

0.015645	0.968709
0.039113	0.525207
0.529552	0.031291
0.976532	0.023468
alpha now = 0.060000

Testing
1,0,0,1 BMU category 2		distance1 1.251460		distance2 1.231415
0,1,1,0 BMU category 1		distance1 2.098484		distance2 2.102774
1,0,1,0 BMU category 2		distance1 2.145420		distance2 1.215770
0,1,0,1 BMU category 1		distance1 1.204523		distance2 2.118420



*/
