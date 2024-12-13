#ifndef CONES_H
#define CONES_H

#include <vector>
#include <string>
#include <functional>
#include <numbers>

typedef std::vector<std::vector<double>> vectorRow;

class Cones {
private:
    vectorRow blue_cones;
    vectorRow yellow_cones;
    vectorRow orange_cones;

    std::string reprCones(const vectorRow& cones) const;

public:
    Cones() = default;

    void addBlueCone(double x, double y, double z);
    void addYellowCone(double x, double y, double z);
    void addOrangeCone(double x, double y, double z);

    void addCones(const Cones& other);

    const vectorRow& getBlueCones() const;
    const vectorRow& getYellowCones() const;

    void map(const std::function<std::vector<double>(const std::vector<double>&)>& mapper);

    void supplementCones();

    std::string toString() const;

    size_t size() const;

    Cones copy() const;

    vectorRow augmentDatasetCircle(vectorRow X, int deg, int radius);
    Cones augmentConesCircle(const Cones& cones, int deg = 20, double radius = 2.0);

    struct ConeData {
        vectorRow blue_cones;
        vectorRow yellow_cones;
        vectorRow orange_cones;
    };

    ConeData toStruct() const;

    Cones fromStruct(const ConeData& data);

    std::pair<vectorRow, std::vector<double>> conesToXY(const Cones& cones);
};

#endif
