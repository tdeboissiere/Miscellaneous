

/////////////////////////
// Vectors
/////////////////////////
#include <vector>
std::vector<int> myvector (3,100);
std::vector<int>::iterator it;

  it = myvector.end();
  it = myvector.insert ( it , 200 );
  // Inset 200 at the end of the vector

  std::cout << "myvector contains:";
  for (it=myvector.begin(); it<myvector.end(); it++)
    std::cout << ' ' << *it;
    // Access vector data
    
/////////////////////////
// Paires Vecteurs
/////////////////////////
#include <utility>
#include <vector>
//Pair of int
vector< pair<int, int> > a;
//Add new entries
a.push_back(std::make_pair(1,2));


/////////////////////////
// Sort
/////////////////////////
#include <algorithm>		
// Define a new sort function
bool myfunction (pair<int,int> i,pair<int,int> j) { return (i.first<j.first); }
// Sort the vector
std::sort(a.begin(),a.end(),myfunction);


/////////////////////////
// Dictionnary class and iterator
/////////////////////////
#include <iostream>
#include <map>

typedef std::map<std::pair<int, int>, int> Dict;
typedef Dict::const_iterator It;

int main()
{
   Dict d;

   d[std::make_pair(0, 0)] = 0;
   d[std::make_pair(1, 2)] = 1;
   d[std::make_pair(2, 1)] = 2;
   d[std::make_pair(2, 3)] = 3;
   d[std::make_pair(3, 2)] = 4;

   for (It it(d.begin()); it != d.end(); ++it)
   {
      int i(it->first.first);
      int j(it->first.second);
      std::cout <<it->second <<' '
                <<d[std::make_pair(j, i)] <<'\n';
   }
}


/////////////////////////
// Pointers and returning an array from a function
/////////////////////////

// in the function 
double * tabl() {
double * mean= new double[4];
return mean;
}

double * array = tabl()  // function call
delete[] array // memory management


/////////////////////////
// Precision for output file doubles/float
/////////////////////////
file.precision(2);
file.setf(ios::fixed);
file.setf(ios::showpoint); 

/////////////////////////
// Dump the first line of a file
/////////////////////////
string dummyLine;		
getline(file2, dummyLine);	//store the first line
//The start reading the file normally


/////////////////////////
// Read file even without knowing number of lines
///////////////////////// 
   ifstream file(file_name);
   while (file >> blabla) {
    // do things
  }  
  

/////////////////////////
// Command line argument
///////////////////////// 
int main(int argc, char *argv[])
{

    TString bolo_name = "FID837";
    if (argc >=2) get_trigged_trace_tree(bolo_name, argv[1]);
    return 0;
    // argc = size of argv
    // argc toujours >=1 argv[0] = file name
}


/////////////////////////
// Look for pattern in txt file
/////////////////////////
        string tmpstr;
        string pattern = "neutron";
        vector <TString> vec_str;
        ifstream feffin(file_name);
        while (getline(feffin, tmpstr))
        {
            size_t found = tmpstr.find(pattern);
            if (found!=string::npos) // do something;
            else vec_str.push_back(TString(tmpstr));
        }
        feffin.close();
        ofstream feffout(file_name);
        for (int u = 0; u<vec_str.size(); u++) {feffout << vec_str[u] << endl;}
        feffout.close();