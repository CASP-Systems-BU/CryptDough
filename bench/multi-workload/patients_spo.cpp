#include "cdough.h"

using namespace cdough::debug;
using namespace cdough::service;

using namespace COMPILED_MPC_PROTOCOL_NAMESPACE;

using T = int64_t;

#define SPO2_THRESHOLD 92
#define POS_THRESHOLD 0.8

// Custom VGG model for Xrays with 3 output classes: COVID, PNEUMONIA, NORMAL
// TODO: move to a models header file.

using HW = cdough::matrix::HeightWidth;

template <typename Engine>
SecureMatrix<T> run_vgg16_xray(Engine& engine, size_t test_size) {
    // Test Data
    const auto inputSize = HW(224, 224);
    const size_t channels_num = 3;
    const size_t precision = 16;

    auto inputPlain = PlainMatrix<T>::RandomMatrix(engine, test_size * inputSize.first,
                                                          channels_num * inputSize.second);
    auto input = engine.secret_share_matrix(inputPlain, 0);
    input.setPrecision(precision);

    // VGG16
    cdough::operators::ml::ModelML<T, SecureMatrix, Engine> model(engine, precision);

    // our vgg16-imagenet model without the last layer, which we replace
    // we a smaller dense layers to output 3 classes.
    model.conv2DLayer(HW(224, 224), 3, 64, HW(3, 3), HW(1, 1), HW(1, 1));
    model.reLULayer(3211264);
    model.conv2DLayer(HW(224, 224), 64, 64, HW(3, 3), HW(1, 1), HW(1, 1));
    model.avgPoolingLayer(HW(224, 224), 64, HW(2, 2), HW(2, 2), HW(0, 0));
    model.reLULayer(802816);
    model.conv2DLayer(HW(112, 112), 64, 128, HW(3, 3), HW(1, 1), HW(1, 1));
    model.reLULayer(1605632);
    model.conv2DLayer(HW(112, 112), 128, 128, HW(3, 3), HW(1, 1), HW(1, 1));
    model.avgPoolingLayer(HW(112, 112), 128, HW(2, 2), HW(2, 2), HW(0, 0));
    model.reLULayer(401408);
    model.conv2DLayer(HW(56, 56), 128, 256, HW(3, 3), HW(1, 1), HW(1, 1));
    model.reLULayer(802816);
    model.conv2DLayer(HW(56, 56), 256, 256, HW(3, 3), HW(1, 1), HW(1, 1));
    model.reLULayer(802816);
    model.conv2DLayer(HW(56, 56), 256, 256, HW(3, 3), HW(1, 1), HW(1, 1));
    model.avgPoolingLayer(HW(56, 56), 256, HW(2, 2), HW(2, 2), HW(0, 0));
    model.reLULayer(200704);
    model.conv2DLayer(HW(28, 28), 256, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    model.reLULayer(401408);
    model.conv2DLayer(HW(28, 28), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    model.reLULayer(401408);
    model.conv2DLayer(HW(28, 28), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    model.avgPoolingLayer(HW(28, 28), 512, HW(2, 2), HW(2, 2), HW(0, 0));
    model.reLULayer(100352);
    model.conv2DLayer(HW(14, 14), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    model.reLULayer(100352);
    model.conv2DLayer(HW(14, 14), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    model.reLULayer(100352);
    model.conv2DLayer(HW(14, 14), 512, 512, HW(3, 3), HW(1, 1), HW(1, 1));
    model.avgPoolingLayer(HW(14, 14), 512, HW(2, 2), HW(2, 2), HW(0, 0));
    model.reLULayer(25088);

    // Starting here: the fully connected layers tailored for Xray classification (3 classes)
    model.avgPoolingLayer(HW(7, 7), 512, HW(2, 2), HW(2, 2), HW(0, 0));
    // "layer": "relu",
    // "input_dim": 512
    model.reLULayer(512);

    // "layer": "fc",
    // "input_dim": 512,
    // "output_dim": 256
    model.fullyConnectedLayer(512, 256);
    // "layer": "relu",
    // "input_dim": 256
    model.reLULayer(256);

    // "layer": "fc",
    // "input_dim": 256,
    // "output_dim": 256
    model.fullyConnectedLayer(256, 256);
    // "layer": "relu",
    // "input_dim": 256
    model.reLULayer(256);

    // "layer": "fc",
    // "input_dim": 256,
    // "output_dim": 3
    model.fullyConnectedLayer(256, 3);
    // "layer": "relu",
    // "input_dim": 3
    model.reLULayer(3);

    // Forward pass
    auto output = model.forward(input);
    return output;
}

int main(int argc, char** argv) {
    EngineRef engine = cdough_init(argc, argv);
    auto pID = engine.getPartyID();

    // TODO: figure out input args
    int ts_test_size = 16384; // engine.getArg<int>("ts-test-size", "r1", 16);
    int rel_test_size = 1024; // engine.getArg<int>("rel-test-size", "r2", 8);
    int img_test_size = 64; //engine.getArg<int>("img-test-size", "r3", 4);

    using A = ASharedVector<T>;
    using B = BSharedVector<T>;

    // Time series schema
    std::vector<std::string> ts_schema = {"[TIMESTAMP]", "[PATIENT_ID]", "[SPO_2]", "SPO_2"};
    // Relational schema. ASTHMA_GROUP is "0: control, 1: treatment"
    std::vector<std::string> rel_schema = {"[PATIENT_ID]", "[ASTHMA_GROUP]"};
    // Classification output schema
    std::vector<std::string> class_schema = {"[PATIENT_ID]", "PROB_COVID", "PROB_PNEUM", "PROB_NORMAL"};

    std::vector<cdough::Vector<T>> spo2_data(ts_schema.size(), Vector<T>(ts_test_size));
    EncodedTable<T> spo2_table = engine.secret_share_table(spo2_data, ts_schema);

    std::vector<cdough::Vector<T>> patient_data(rel_schema.size(), Vector<T>(rel_test_size));
    EncodedTable<T> patient_table = engine.secret_share_table(patient_data, rel_schema);

    std::vector<cdough::Vector<T>> class_data(class_schema.size(), Vector<T>(img_test_size));
    EncodedTable<T> class_table = engine.secret_share_table(class_data, class_schema);

    // start timer
    stopwatch::timepoint("Start");
    // STEP 1: call vgg16 model on images and get classification data that we insert to class_table
    SecureMatrix<T> classif_res = run_vgg16_xray(engine, img_test_size);

    // Extract probablilities as 3 columns and insert into class_table
    A classif_data = classif_res.getContents();

    // TODO: add helper functions in matrix.h to extract columns
    // TODO: no better way to insert data into table columns?
    (*class_table["PROB_COVID"].contents.get()) = classif_data.subset(0, 3);
    (*class_table["PROB_PNEUM"].contents.get()) = classif_data.subset(1, 3);
    (*class_table["PROB_NORMAL"].contents.get()) = classif_data.subset(2, 3);

    stopwatch::timepoint("Stage 1: inference");

    // STEP 2: Join class_table with patient_table and append asthma column
    auto patients_with_asthma_and_class = patient_table.inner_join(class_table, 
        {"[PATIENT_ID]"}, 
        {
            {"[ASTHMA_GROUP]", "[ASTHMA_GROUP]", copy<B>},
            {"PROB_COVID", "PROB_COVID", copy<A>},
            {"PROB_PNEUM", "PROB_PNEUM", copy<A>},
            {"PROB_NORMAL", "PROB_NORMAL", copy<A>}
        });
    
    stopwatch::timepoint("Stage 2: Relational");
    
    // STEP 3: Compute the number of hypoxemia incidents per patient, 
    // where a hypoxemia incident is a period (session window) where the patient's SPO_2 was below 92. 
    // Output PatientIncidents: <PatientID, IncidentCount>

    spo2_table.sort({"[PATIENT_ID]", "[TIMESTAMP]"}, ASC);

    // flipping because threshold window computes session by checking
    // if a value exceeds a threshold but we want to compute if it's below
    spo2_table["SPO_2"] = (-spo2_table["SPO_2"] + SPO2_THRESHOLD);
    spo2_table.convert_a2b("SPO_2", "[SPO_2]");

    spo2_table.addColumns({"[INCIDENT_ID]"}, spo2_table.size());

    spo2_table.threshold_session_window({"[PATIENT_ID]"}, "[SPO_2]", "[TIMESTAMP]",
                                           "[INCIDENT_ID]", 0, false);
    
    spo2_table.filter(spo2_table["[INCIDENT_ID]"] > 0);
    spo2_table.distinct({"[PATIENT_ID]", "[INCIDENT_ID]"});

    spo2_table.addColumns({"INCIDENT_CNT", "[INCIDENT_CNT]"}, spo2_table.size());

    spo2_table.aggregate({"[PATIENT_ID]"}, {{"INCIDENT_CNT", "INCIDENT_CNT", count<A>}},
    {.mark_valid = true});

    // Remove unecessary columns
    spo2_table.deleteColumns({"SPO_2", "[SPO_2]", "[TIMESTAMP]", "[INCIDENT_ID]"});

    stopwatch::timepoint("Stage 3: Time series");

    // STEP 4: Find patients with IncidentCount>0 by joining the 
    // PatientsWithAsthmaAndClass table with PatientIncidents, 
    // group by Asthma, and compute average per probability column.
    // Output Result = <AsthmaGroup, ProbAvg1, ProbAvg2, ProbAvg3>
    spo2_table.convert_a2b("INCIDENT_CNT", "[INCIDENT_CNT]");
    spo2_table.filter(spo2_table["[INCIDENT_CNT]"] > 0); // patients with at least one incident

    // We select the patients using a join between `patients_with_asthma_and_class` and `spo2_table`.
    auto result = spo2_table.inner_join(patients_with_asthma_and_class, 
        {"[PATIENT_ID]"}, 
        {
            {"[ASTHMA_GROUP]", "[ASTHMA_GROUP]", copy<B>},
            {"PROB_COVID", "PROB_COVID", copy<A>},
            {"PROB_PNEUM", "PROB_PNEUM", copy<A>},
            {"PROB_NORMAL", "PROB_NORMAL", copy<A>}
        });

    result.addColumns({"CNT_COVID", "CNT_PNEUM", "CNT_NORMAL"}, result.size());
    
    result["CNT_COVID"] = result["PROB_COVID"] > POS_THRESHOLD;
    result["CNT_PNEUM"] = result["PROB_PNEUM"] > POS_THRESHOLD;
    result["CNT_NORMAL"] = result["PROB_NORMAL"] > POS_THRESHOLD;
    
    result.aggregate(
        {"[ASTHMA_GROUP]"},
        {
            {"CNT_COVID", "CNT_COVID", count<A>},
            {"CNT_PNEUM", "CNT_PNEUM", count<A>},
            {"CNT_NORMAL", "CNT_NORMAL", count<A>},
        });

    // Remove unecessary columns
    // TODO: better use project()
    result.deleteColumns({"PROB_COVID", "PROB_PNEUM", "PROB_NORMAL"});

    stopwatch::timepoint("Stage 4: Relational");

    auto output = result.open_with_schema(false);
    print_table(output, pID);

    return 0;
}
