#pragma once

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "core/communication/ring.h"

#if defined(CDOUGH_ENABLE_TLS)
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/ssl.h>
#include <openssl/x509.h>
#endif

/**
 * @file network_utils.h
 * @brief Cross-platform networking helpers for the no-copy communicator.
 *
 * All party-to-party traffic goes through the `Conn` class below. `Conn` owns the
 * socket descriptor privately, so there is no way to call a bare `recv()`/`send()`
 * on a connection: every byte is funnelled through `send_all`/`read_some`. That is
 * deliberate. Before TLS existed, several call sites in `no_copy_communicator.h`
 * used the raw descriptor directly, and a TLS retrofit that converted only the
 * wrapper functions would have compiled, run, and silently carried most MPC traffic
 * in plaintext.
 *
 * When built with `-DTLS=ON` (which defines `CDOUGH_ENABLE_TLS`), every connection
 * is a mutually-authenticated TLS 1.3 session. Otherwise `Conn` is a thin wrapper
 * over the descriptor and behaviour is byte-for-byte what it was before.
 */

#if defined(CDOUGH_ENABLE_TLS)

namespace cdough::tls {

/**
 * @brief Read an environment variable, throwing if it is unset or empty.
 *
 * TLS material is configured the same way the rest of the launcher is configured
 * (see startmpc.h), so that no repository file needs to learn about certificates.
 */
inline std::string require_env(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        throw std::runtime_error(
            std::string("TLS is enabled but ") + name +
            " is not set. Expected: CDOUGH_TLS_CERT (this party's certificate), "
            "CDOUGH_TLS_KEY (its private key), CDOUGH_TLS_PEER_CERTS (comma-separated "
            "peer certificates to pin).");
    }
    return std::string(value);
}

inline std::string openssl_error() {
    unsigned long code = ERR_get_error();
    if (code == 0) return "no OpenSSL error queued";
    char buffer[256] = {0};
    ERR_error_string_n(code, buffer, sizeof(buffer));
    return std::string(buffer);
}

/** @brief Lowercase hex SHA-256 of a certificate's DER encoding. */
inline std::string fingerprint(X509* certificate) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_length = 0;
    if (X509_digest(certificate, EVP_sha256(), digest, &digest_length) != 1) {
        throw std::runtime_error("Could not compute certificate fingerprint");
    }
    std::ostringstream out;
    for (unsigned int i = 0; i < digest_length; ++i) {
        out << std::hex << (digest[i] >> 4) << (digest[i] & 0x0f);
    }
    return out.str();
}

inline std::vector<std::string> split_list(const std::string& value, char delimiter) {
    std::vector<std::string> items;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, delimiter)) {
        if (!item.empty()) items.push_back(item);
    }
    return items;
}

/**
 * @brief The set of peer certificate fingerprints this party will accept.
 *
 * Pinning rather than a CA: with a small, fixed set of mutually-distrusting parties
 * there is no third party everyone would agree to trust, and the certificates can be
 * exchanged out of band once. Fingerprints are compared after the handshake and
 * before any application data is exchanged.
 */
inline const std::vector<std::string>& pinned_fingerprints() {
    static const std::vector<std::string> pinned = [] {
        std::vector<std::string> results;
        for (const auto& path : split_list(require_env("CDOUGH_TLS_PEER_CERTS"), ',')) {
            FILE* file = std::fopen(path.c_str(), "r");
            if (file == nullptr) {
                throw std::runtime_error("Could not open pinned peer certificate: " + path);
            }
            // A peer file may hold more than one certificate.
            bool found_any = false;
            while (X509* certificate = PEM_read_X509(file, nullptr, nullptr, nullptr)) {
                results.push_back(fingerprint(certificate));
                X509_free(certificate);
                found_any = true;
            }
            std::fclose(file);
            if (!found_any) {
                throw std::runtime_error("No certificate found in pinned peer file: " + path);
            }
        }
        if (results.empty()) {
            throw std::runtime_error("CDOUGH_TLS_PEER_CERTS listed no certificates");
        }
        return results;
    }();
    return pinned;
}

/**
 * @brief One process-wide SSL_CTX, shared by every connection thread.
 *
 * OpenSSL >= 1.1 makes a shared SSL_CTX safe to use concurrently; each connection
 * still gets its own SSL object. TLS 1.3 is required, which gives a 1-RTT handshake
 * and forward secrecy without further configuration.
 */
inline SSL_CTX* context() {
    static SSL_CTX* ctx = [] {
        SSL_CTX* created = SSL_CTX_new(TLS_method());
        if (created == nullptr) {
            throw std::runtime_error("SSL_CTX_new failed: " + openssl_error());
        }
        if (SSL_CTX_set_min_proto_version(created, TLS1_3_VERSION) != 1) {
            throw std::runtime_error("Could not require TLS 1.3: " + openssl_error());
        }

        const std::string cert_path = require_env("CDOUGH_TLS_CERT");
        const std::string key_path = require_env("CDOUGH_TLS_KEY");

        if (SSL_CTX_use_certificate_file(created, cert_path.c_str(), SSL_FILETYPE_PEM) != 1) {
            throw std::runtime_error("Could not load certificate " + cert_path + ": " +
                                     openssl_error());
        }
        if (SSL_CTX_use_PrivateKey_file(created, key_path.c_str(), SSL_FILETYPE_PEM) != 1) {
            throw std::runtime_error("Could not load private key " + key_path + ": " +
                                     openssl_error());
        }
        if (SSL_CTX_check_private_key(created) != 1) {
            throw std::runtime_error("Private key does not match certificate: " + openssl_error());
        }

        // Demand a certificate from the peer in both directions. The certificates are
        // self-signed, so chain construction is expected to fail; the callback lets the
        // handshake proceed and the fingerprint check below is what actually authorises
        // the peer. Without SSL_VERIFY_FAIL_IF_NO_PEER_CERT a server would accept an
        // anonymous client.
        SSL_CTX_set_verify(created, SSL_VERIFY_PEER | SSL_VERIFY_FAIL_IF_NO_PEER_CERT,
                           [](int, X509_STORE_CTX*) { return 1; });

        // Force the pinned set to load now, so a misconfiguration is a startup error
        // rather than a confusing mid-handshake failure.
        (void)pinned_fingerprints();
        return created;
    }();
    return ctx;
}

/**
 * @brief Authorise a peer by certificate fingerprint. Fails closed.
 *
 * Called immediately after the handshake and before any application data. This is
 * the step that gives pinning its value; the handshake alone proves only that the
 * peer holds a private key, not that it is one of our parties.
 */
inline void verify_pinned_peer(SSL* ssl, const std::string& context_description) {
    X509* peer = SSL_get1_peer_certificate(ssl);
    if (peer == nullptr) {
        throw std::runtime_error("Peer presented no certificate (" + context_description + ")");
    }
    std::string peer_fingerprint;
    try {
        peer_fingerprint = fingerprint(peer);
    } catch (...) {
        X509_free(peer);
        throw;
    }
    X509_free(peer);

    for (const auto& allowed : pinned_fingerprints()) {
        if (allowed == peer_fingerprint) return;
    }
    throw std::runtime_error("Peer certificate is not pinned (" + context_description +
                             "); SHA-256 " + peer_fingerprint +
                             ". Refusing the connection rather than falling back.");
}

}  // namespace cdough::tls

#endif  // CDOUGH_ENABLE_TLS

/**
 * @brief One party-to-party connection.
 *
 * The descriptor is private on purpose; see the file comment. Copyable, because the
 * socket maps built in startmpc.h are handed around by value. Ownership is not
 * reference counted: `close_conn` is called exactly once per connection by
 * `NoCopyCommunicator`'s destructor, matching the pre-TLS `close(sockfd)` behaviour.
 */
class Conn {
  public:
    Conn() = default;

    /** @brief Adopt a connected descriptor with no TLS (plaintext transport). */
    static Conn plain(int fd) {
        Conn conn;
        conn.fd_ = fd;
        return conn;
    }

#if defined(CDOUGH_ENABLE_TLS)
    /** @brief Adopt a connected descriptor and run the TLS client handshake. */
    static Conn client(int fd, const std::string& description) {
        return handshake(fd, description, /*is_server=*/false);
    }

    /** @brief Adopt an accepted descriptor and run the TLS server handshake. */
    static Conn server(int fd, const std::string& description) {
        return handshake(fd, description, /*is_server=*/true);
    }
#endif

    bool valid() const { return fd_ >= 0; }

    /** @brief Send exactly `buf_size` bytes, looping over short writes. */
    size_t send_all(const char* buf, ssize_t buf_size) {
        ssize_t bytes_sent = 0;
        while (bytes_sent < buf_size) {
            ssize_t ret = write_some(buf + bytes_sent, buf_size - bytes_sent);
            if (ret <= 0) {
                throw std::runtime_error("Conn::send_all: send failed");
            }
            bytes_sent += ret;
        }
        return static_cast<size_t>(bytes_sent);
    }

    /**
     * @brief Read up to `buf_size` bytes; returns the count, 0 on clean shutdown,
     * or -1 on error. Callers loop over this exactly as they looped over `recv`.
     */
    ssize_t read_some(char* buf, ssize_t buf_size) {
#if defined(CDOUGH_ENABLE_TLS)
        if (ssl_ != nullptr) {
            // Loop rather than recurse: WANT_READ/WANT_WRITE can repeat (a TLS 1.3
            // session ticket or key update arriving mid-stream), and recursion here
            // would grow the stack once per occurrence.
            for (;;) {
                int ret = SSL_read(ssl_, buf, static_cast<int>(buf_size));
                if (ret > 0) return ret;
                int reason = SSL_get_error(ssl_, ret);
                if (reason == SSL_ERROR_ZERO_RETURN) return 0;  // peer closed cleanly
                if (reason != SSL_ERROR_WANT_READ && reason != SSL_ERROR_WANT_WRITE) {
                    return -1;
                }
                // Blocking socket: retry.
            }
        }
#endif
        return recv(fd_, buf, buf_size, 0);
    }

    /** @brief Read exactly `buf_size` bytes, looping over short reads. */
    ssize_t read_exact(char* buf, ssize_t buf_size) {
        ssize_t bytes_received = 0;
        while (bytes_received < buf_size) {
            ssize_t ret = read_some(buf + bytes_received, buf_size - bytes_received);
            if (ret == 0) return -1;  // connection closed before the message completed
            if (ret < 0) throw std::runtime_error("Conn::read_exact: recv failed");
            bytes_received += ret;
        }
        return bytes_received;
    }

    void close_conn() {
        if (fd_ < 0) return;
#if defined(CDOUGH_ENABLE_TLS)
        if (ssl_ != nullptr) {
            SSL_shutdown(ssl_);
            SSL_free(ssl_);
            ssl_ = nullptr;
        }
#endif
        close(fd_);
        fd_ = -1;
    }

  private:
    ssize_t write_some(const char* buf, ssize_t buf_size) {
#if defined(CDOUGH_ENABLE_TLS)
        if (ssl_ != nullptr) {
            for (;;) {  // see read_some for why this loops rather than recurses
                int ret = SSL_write(ssl_, buf, static_cast<int>(buf_size));
                if (ret > 0) return ret;
                int reason = SSL_get_error(ssl_, ret);
                if (reason != SSL_ERROR_WANT_READ && reason != SSL_ERROR_WANT_WRITE) {
                    return -1;
                }
            }
        }
#endif
        return send(fd_, buf, buf_size, 0);
    }

#if defined(CDOUGH_ENABLE_TLS)
    static Conn handshake(int fd, const std::string& description, bool is_server) {
        Conn conn;
        conn.fd_ = fd;
        conn.ssl_ = SSL_new(cdough::tls::context());
        if (conn.ssl_ == nullptr) {
            close(fd);
            throw std::runtime_error("SSL_new failed: " + cdough::tls::openssl_error());
        }
        if (SSL_set_fd(conn.ssl_, fd) != 1) {
            SSL_free(conn.ssl_);
            close(fd);
            throw std::runtime_error("SSL_set_fd failed: " + cdough::tls::openssl_error());
        }
        int ret = is_server ? SSL_accept(conn.ssl_) : SSL_connect(conn.ssl_);
        if (ret != 1) {
            std::string reason = cdough::tls::openssl_error();
            SSL_free(conn.ssl_);
            close(fd);
            throw std::runtime_error(std::string("TLS handshake failed as ") +
                                     (is_server ? "server" : "client") + " (" + description +
                                     "): " + reason);
        }
        // Authorise before a single byte of application data moves.
        cdough::tls::verify_pinned_peer(conn.ssl_, description);
        return conn;
    }

    SSL* ssl_ = nullptr;
#endif

    int fd_ = -1;
};

inline int socket_create(int port) {
    int server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock == -1) {
        perror("socket failed");
        throw std::runtime_error("Socket failed");
    }

    int opt = 1;
    if (setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("setsockopt failed");
        close(server_sock);
        throw std::runtime_error("setsockopt failed");
    }

    struct sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    // Attempt to bind; on failure, try again after 1 minute
    // This may help recover from intermittent socket issues.
    if (bind(server_sock, (struct sockaddr*)&address, sizeof(address)) == -1) {
        auto m = "bind failed on port " + std::to_string(port) + ": trying again after 60 sec";
        perror(m.c_str());
        sleep(60);
        if (bind(server_sock, (struct sockaddr*)&address, sizeof(address)) == -1) {
            close(server_sock);
            throw std::runtime_error("Bind failed twice, exiting!");
        }
        std::cout << "...OK!\n";
    }

    socklen_t addrlen = sizeof(address);
    if (getsockname(server_sock, (struct sockaddr*)&address, &addrlen) == -1) {
        perror("getsockname failed");
        close(server_sock);
        throw std::runtime_error("getsockname failed");
    }

    char ipstr[INET_ADDRSTRLEN] = {0};
    if (inet_ntop(AF_INET, &address.sin_addr, ipstr, sizeof(ipstr)) == NULL) {
        perror("inet_ntop failed");
        close(server_sock);
        throw std::runtime_error("inet_ntop failed");
    }

    return server_sock;
}

/**
 * @brief Connect to a peer, retrying briefly while it comes up.
 *
 * The peer may not be listening yet. Listeners block in `accept()` indefinitely, so
 * only the connecting side can time out; that asymmetry is why parties are started in
 * descending rank order (party i only ever connects to j > i). The retry budget below
 * is generous enough to also tolerate a peer restarting mid-deployment, which matters
 * when the parties are separate organizations launching independently.
 */
inline int socket_connect(const std::string& hostname, int port) {
    struct addrinfo hints{}, *res;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(port);
    int status = getaddrinfo(hostname.c_str(), port_str.c_str(), &hints, &res);
    if (status != 0) throw std::runtime_error("Failed to resolve hostname: " + hostname);

    const char* retry_env = std::getenv("CDOUGH_CONNECT_RETRIES");
    const int max_retries = retry_env ? std::atoi(retry_env) : 3;

    for (int attempts = 0;; ++attempts) {
        // A fresh descriptor per attempt: a failed connect() leaves the socket in an
        // unusable state, so retrying on the same descriptor silently never succeeds.
        int sockfd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (sockfd < 0) {
            freeaddrinfo(res);
            throw std::runtime_error("Failed to create socket");
        }

        int opt = 1;
        if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            close(sockfd);
            freeaddrinfo(res);
            throw std::runtime_error("Could not set SO_REUSEADDR");
        }

        if (connect(sockfd, res->ai_addr, res->ai_addrlen) == 0) {
            freeaddrinfo(res);
            return sockfd;
        }
        close(sockfd);

        if (attempts >= max_retries) {
            freeaddrinfo(res);
            throw std::runtime_error("Failed to connect to server (" + hostname + ":" +
                                     std::to_string(port) + ") after " +
                                     std::to_string(max_retries) + " attempts");
        }
        sleep(1);
    }
}

inline int send_meta(Conn& conn, int byte_count) {
    conn.send_all(reinterpret_cast<const char*>(&byte_count), sizeof(byte_count));
    return 1;
}

inline int recv_meta(Conn& conn) {
    int byte_count = 0;
    if (conn.read_exact(reinterpret_cast<char*>(&byte_count), sizeof(byte_count)) < 0) {
        return -1;  // Break connection
    }
    return byte_count;
}

inline int send_message(Conn& conn, const RingEntry& entry) {
    conn.send_all(entry.buffer, entry.used);
    return 1;
}

inline size_t send_wrapper(Conn& conn, const char* buf, ssize_t buf_size) {
    return conn.send_all(buf, buf_size);
}

inline int recv_message(Conn& conn, char* buf, ssize_t buf_size) {
    return static_cast<int>(conn.read_exact(buf, buf_size));
}
