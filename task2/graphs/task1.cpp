#include <vector>

#include "pbPlots.hpp"
#include "supportLib.hpp"

#define SERIAL_TIME_20000 0.63
#define SERIAL_TIME_40000 2.49


void render_graph(std::vector<double> &x, std::vector<double> &y, std::string &filename) {
    ScatterPlotSeries *series = GetDefaultScatterPlotSeriesSettings();
	series->xs = &x;
	series->ys = &y;
	series->linearInterpolation = true;
	series->lineType = toVector(L"solid");
    series->pointType = toVector(L"dots");
	series->lineThickness = 1;

    ScatterPlotSettings *settings = GetDefaultScatterPlotSettings();
    settings->width = 800;
	settings->height = 600;
    settings->autoBoundaries = true;
	settings->autoPadding = true;
    settings->xLabel = toVector(L"P");
	settings->yLabel = toVector(L"Sp");
    settings->scatterPlotSeries->push_back(series);

    RGBABitmapImageReference *imageRef = CreateRGBABitmapImageReference();
    StringReference *errorMessage = CreateStringReferenceLengthValue(0, L' ');

    bool result = DrawScatterPlotFromSettings(imageRef, settings, errorMessage);

    if (result) {
        std::vector<double> *pngData = ConvertToPNG(imageRef->image);
        WriteToFile(pngData, filename);
        DeleteImage(imageRef->image);
    } else {
        std::cerr << "Error: ";
		for(wchar_t c : *errorMessage->string){
			std::wcerr << c;
		}
		std::cerr << std::endl;
    }

    FreeAllocations();
}

int main() {
    std::vector<double> num_threads{1, 2, 4, 7, 8, 16, 20, 40};
    
    std::vector<double> t_20{0.62, 0.32, 0.16, 0.09, 0.08, 0.04, 0.03, 0.02};
    std::vector<double> t_40{2.24, 1.44, 1.24, 0.41, 0.35, 0.16, 0.13, 0.07};
    
    std::vector <double> y_20{};
    std::vector <double> y_40{};
    
    for (int i = 0; i < t_20.size(); i++)
        y_20.push_back(SERIAL_TIME_20000 / t_20.at(i));

    for (int i = 0; i < t_40.size(); i++)
        y_40.push_back(SERIAL_TIME_40000 / t_40.at(i));

    std::string task1_20_filename = "./task1_20000.png";
    std::string task1_40_filename = "./task1_40000.png";
    
    render_graph(num_threads, y_20, task1_20_filename);
    render_graph(num_threads, y_40, task1_40_filename);
}
