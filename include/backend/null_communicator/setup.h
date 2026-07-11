#pragma once

#include <set>

#include "backend/common/runtime.h"
#include "mpc.h"

namespace cdough::service::null_service {

namespace {
    template <typename PRG = PRGAlgorithm, typename PermGen = HMShardedPermutationGenerator>
    auto setup_random_generation() {
        auto commonPRGManager = std::make_shared<CommonPRGManager>(1);

        // Setup dummy common PRG. Use seed of all zeros.
        auto local_key = std::vector<unsigned char>(crypto_aead_aes256gcm_KEYBYTES);
        cdough::random::AESPRGAlgorithm::aesKeyGen(local_key);
        std::unique_ptr<DeterministicPRGAlgorithm> prg_algorithm =
            std::make_unique<cdough::random::AESPRGAlgorithm>(local_key);
        auto commonPRG = std::make_shared<cdough::random::CommonPRG>(std::move(prg_algorithm), 0);

        // This local PRG applies to relative party 0 (ourself) as well as the group of one party
        // (also ourself).
        commonPRGManager->add(commonPRG, 0);
        commonPRGManager->add(commonPRG, std::set<int>({0}));

        std::vector<std::set<int>> groups;
        groups.push_back({0});

        // setup the Permutation Generator
        // For confidential mode, this is not used
        auto sharded_generator = std::make_shared<PermGen>(0, commonPRGManager, groups);

        // This will always return 0.
        auto zeroSharingGenerator = std::make_shared<ZeroSharingGenerator>(1, commonPRGManager);

        return std::make_unique<RandomnessManager>(commonPRGManager, zeroSharingGenerator,
                                                   CorrRegistry_t{}, sharded_generator);
    }
}  // namespace

namespace plaintext_1pc {
    init_mpc_types(cdough::Vector, std::vector, cdough::EVector, 1);
    init_mpc_system(cdough::NullCommunicator, cdough::Plaintext_1PC_Factory);

    static EngineRef cdough_init(int argc, char** argv) {
        oc::CLP cmd(argc, argv);

        auto threads_num = register_cli<int>(cmd, "threads", "t", 1);
        auto batch_size = register_cli<int>(cmd, "batch", "b", DEFAULT_BATCH_SIZE);
        auto setting = parse_setting(register_cli<std::string>(cmd, "setting", "s", "same"));
        auto host_prefix = register_cli<std::string>(cmd, "host-prefix", "x", "localhost");

        if constexpr (CONFIDENTIAL_1PC) {
            std::cout << "[cdough] Running Confidential 1PC\n";
        }

        engine_ptr.push_back(std::make_unique<RunTime<ProtocolFactory>>(batch_size, threads_num, cmd));
        auto& engine = *engine_ptr.back();

        cdough::benchmarking::stopwatch::partyID = 0;
        engine.setup_workers(0);

        ProtocolFactory protocolFactory(0, 1);

        if (cmd.isSet("help") || cmd.isSet("h")) {
            usage(argv, 0);
        }

        for (int i = 0; i < engine.get_num_threads(); ++i) {
            // create a null communicator
            engine.workers[i].attach(std::make_unique<cdough::NullCommunicator>(host_prefix),
                                     setup_random_generation());

            engine.workers[i].init_proto<int8_t>(protocolFactory);
            engine.workers[i].init_proto<int16_t>(protocolFactory);
            engine.workers[i].init_proto<int32_t>(protocolFactory);
            engine.workers[i].init_proto<int64_t>(protocolFactory);
            engine.workers[i].init_proto<__int128_t>(protocolFactory);
        }

        return engine;
    }
}  // namespace plaintext_1pc

namespace dummy_0pc {
    init_mpc_types(cdough::Vector, std::vector, cdough::EVector, 1);
    init_mpc_system(cdough::NullCommunicator, cdough::Dummy_0PC_Factory);

    static EngineRef cdough_init(int argc, char** argv) {
        // We don't use cmd for Dummy 0PC, but cdough programs might.
        oc::CLP cmd(argc, argv);

        // Set Batch Size / Number of Threads
        int threads_num = 1;
        int batch_size = -1;

        engine_ptr.push_back(std::make_shared<RunTime<ProtocolFactory>>(batch_size, threads_num, cmd));
        auto& engine = *engine_ptr.back();

        cdough::benchmarking::stopwatch::partyID = 0;
        engine.setup_workers(0);

        ProtocolFactory protocolFactory(0, 1);

        for (int i = 0; i < engine.get_num_threads(); ++i) {
            // create a null communicator
            // no randomness generation for dummy
            engine.workers[i].attach(
                std::make_unique<cdough::NullCommunicator>(),
                setup_random_generation<ZeroRandomGenerator, ZeroPermutationGenerator>());

            engine.workers[i].init_proto<int8_t>(protocolFactory);
            engine.workers[i].init_proto<int16_t>(protocolFactory);
            engine.workers[i].init_proto<int32_t>(protocolFactory);
            engine.workers[i].init_proto<int64_t>(protocolFactory);
            engine.workers[i].init_proto<__int128_t>(protocolFactory);
        }

        return engine;
    }
}  // namespace dummy_0pc

}  // namespace cdough::service::null_service
