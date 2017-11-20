#include <cerrno>
#include <cstring>
#include <iomanip>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

#include "emu/mmu.h"
#include "emu/state.h"
#include "riscv/abi.h"
#include "riscv/context.h"
#include "util/format.h"

namespace {

// Formatter for escaped strings.
struct Escape_formatter {
    const char* pointer;
    size_t length;
};

Escape_formatter escape(const char *pointer, size_t length) {
    return { pointer, length };
}

std::ostream& operator <<(std::ostream& stream, Escape_formatter helper) {
    // State_saver saver {stream};

    const char *start = helper.pointer;
    const char *end = start + helper.length;

    // These are for escaped characters
    for (const char *pointer = helper.pointer; pointer != end; pointer++) {

        // Skip normal characters.
        if (*pointer != '"' && *pointer != '\\' && isprint(*pointer)) continue;

        // Print out all unprinted normal characters.
        if (pointer != start) stream.write(start, pointer - start);

        switch (*pointer) {
            case '"': stream << "\\\""; break;
            case '\\': stream << "\\\\"; break;
            case '\n': stream << "\\n"; break;
            case '\t': stream << "\\t"; break;
            default: 
                stream << "\\x" << std::setfill('0') << std::setw(2) << std::hex
                       << static_cast<int>(static_cast<unsigned char>(*pointer));
                break;
        }

        start = pointer + 1;
    }

    if (end != start) stream.write(start, end - start);

    return stream;
}

// Format for nullable pointers.
struct Pointer_formatter {
    emu::reg_t value;
};

Pointer_formatter pointer(emu::reg_t value) {
    return { value };
}

std::ostream& operator <<(std::ostream& stream, Pointer_formatter formatter) {
    if (formatter.value) {
        stream << std::hex << std::showbase << formatter.value;
    } else {
        stream << "NULL";
    }
    return stream;
}

/* Converters between guest and host data enums structure */

// Only translate POSIX specified subset of error numbers should be sufficient.
riscv::abi::Errno convert_errno_from_host(int number) {
    switch (number) {
        case E2BIG              : return riscv::abi::Errno::e2big;
        case EACCES             : return riscv::abi::Errno::eacces;
        case EADDRINUSE         : return riscv::abi::Errno::eaddrinuse;
        case EADDRNOTAVAIL      : return riscv::abi::Errno::eaddrnotavail;
        case EAFNOSUPPORT       : return riscv::abi::Errno::eafnosupport;
        case EAGAIN             : return riscv::abi::Errno::eagain;
        case EALREADY           : return riscv::abi::Errno::ealready;
        case EBADF              : return riscv::abi::Errno::ebadf;
        case EBADMSG            : return riscv::abi::Errno::ebadmsg;
        case EBUSY              : return riscv::abi::Errno::ebusy;
        case ECANCELED          : return riscv::abi::Errno::ecanceled;
        case ECHILD             : return riscv::abi::Errno::echild;
        case ECONNABORTED       : return riscv::abi::Errno::econnaborted;
        case ECONNREFUSED       : return riscv::abi::Errno::econnrefused;
        case ECONNRESET         : return riscv::abi::Errno::econnreset;
        case EDEADLK            : return riscv::abi::Errno::edeadlk;
        case EDESTADDRREQ       : return riscv::abi::Errno::edestaddrreq;
        case EDOM               : return riscv::abi::Errno::edom;
        case EDQUOT             : return riscv::abi::Errno::edquot;
        case EEXIST             : return riscv::abi::Errno::eexist;
        case EFAULT             : return riscv::abi::Errno::efault;
        case EFBIG              : return riscv::abi::Errno::efbig;
        case EHOSTUNREACH       : return riscv::abi::Errno::ehostunreach;
        case EIDRM              : return riscv::abi::Errno::eidrm;
        case EILSEQ             : return riscv::abi::Errno::eilseq;
        case EINPROGRESS        : return riscv::abi::Errno::einprogress;
        case EINTR              : return riscv::abi::Errno::eintr;
        case EINVAL             : return riscv::abi::Errno::einval;
        case EIO                : return riscv::abi::Errno::eio;
        case EISCONN            : return riscv::abi::Errno::eisconn;
        case EISDIR             : return riscv::abi::Errno::eisdir;
        case ELOOP              : return riscv::abi::Errno::eloop;
        case EMFILE             : return riscv::abi::Errno::emfile;
        case EMLINK             : return riscv::abi::Errno::emlink;
        case EMSGSIZE           : return riscv::abi::Errno::emsgsize;
        case EMULTIHOP          : return riscv::abi::Errno::emultihop;
        case ENAMETOOLONG       : return riscv::abi::Errno::enametoolong;
        case ENETDOWN           : return riscv::abi::Errno::enetdown;
        case ENETRESET          : return riscv::abi::Errno::enetreset;
        case ENETUNREACH        : return riscv::abi::Errno::enetunreach;
        case ENFILE             : return riscv::abi::Errno::enfile;
        case ENOBUFS            : return riscv::abi::Errno::enobufs;
        case ENODEV             : return riscv::abi::Errno::enodev;
        case ENOENT             : return riscv::abi::Errno::enoent;
        case ENOEXEC            : return riscv::abi::Errno::enoexec;
        case ENOLCK             : return riscv::abi::Errno::enolck;
        case ENOLINK            : return riscv::abi::Errno::enolink;
        case ENOMEM             : return riscv::abi::Errno::enomem;
        case ENOMSG             : return riscv::abi::Errno::enomsg;
        case ENOPROTOOPT        : return riscv::abi::Errno::enoprotoopt;
        case ENOSPC             : return riscv::abi::Errno::enospc;
        case ENOSYS             : return riscv::abi::Errno::enosys;
        case ENOTCONN           : return riscv::abi::Errno::enotconn;
        case ENOTDIR            : return riscv::abi::Errno::enotdir;
        case ENOTEMPTY          : return riscv::abi::Errno::enotempty;
        case ENOTRECOVERABLE    : return riscv::abi::Errno::enotrecoverable;
        case ENOTSOCK           : return riscv::abi::Errno::enotsock;
#if ENOTSUP != EOPNOTSUPP
        case ENOTSUP            : return riscv::abi::Errno::enotsup;
#endif
        case ENOTTY             : return riscv::abi::Errno::enotty;
        case ENXIO              : return riscv::abi::Errno::enxio;
        case EOPNOTSUPP         : return riscv::abi::Errno::eopnotsupp;
        case EOVERFLOW          : return riscv::abi::Errno::eoverflow;
        case EOWNERDEAD         : return riscv::abi::Errno::eownerdead;
        case EPERM              : return riscv::abi::Errno::eperm;
        case EPIPE              : return riscv::abi::Errno::epipe;
        case EPROTO             : return riscv::abi::Errno::eproto;
        case EPROTONOSUPPORT    : return riscv::abi::Errno::eprotonosupport;
        case EPROTOTYPE         : return riscv::abi::Errno::eprototype;
        case ERANGE             : return riscv::abi::Errno::erange;
        case EROFS              : return riscv::abi::Errno::erofs;
        case ESPIPE             : return riscv::abi::Errno::espipe;
        case ESRCH              : return riscv::abi::Errno::esrch;
        case ESTALE             : return riscv::abi::Errno::estale;
        case ETIMEDOUT          : return riscv::abi::Errno::etimedout;
        case ETXTBSY            : return riscv::abi::Errno::etxtbsy;
#if EWOULDBLOCK != EAGAIN
        case EWOULDBLOCK        : return riscv::abi::Errno::ewouldblock;
#endif
        case EXDEV              : return riscv::abi::Errno::exdev;
        default:
            util::log("Fail to translate host errno = {} to guest errno\n", number);
            return riscv::abi::Errno::enosys;
    }
}

void convert_stat_from_host(riscv::abi::stat *guest_stat, struct stat *host_stat) {
    guest_stat->st_dev          = host_stat->st_dev;
    guest_stat->st_ino          = host_stat->st_ino;
    guest_stat->st_mode         = host_stat->st_mode;
    guest_stat->st_nlink        = host_stat->st_nlink;
    guest_stat->st_uid          = host_stat->st_uid;
    guest_stat->st_gid          = host_stat->st_gid;
    guest_stat->st_rdev         = host_stat->st_rdev;
    guest_stat->st_size         = host_stat->st_size;
    guest_stat->st_blocks       = host_stat->st_blocks;
    guest_stat->st_blksize      = host_stat->st_blksize;
    guest_stat->guest_st_atime  = host_stat->st_atime;
    guest_stat->st_atime_nsec   = host_stat->st_atim.tv_nsec;
    guest_stat->guest_st_mtime  = host_stat->st_mtim.tv_sec;
    guest_stat->st_mtime_nsec   = host_stat->st_mtim.tv_nsec;
    guest_stat->guest_st_ctime  = host_stat->st_ctim.tv_sec;
    guest_stat->st_ctime_nsec   = host_stat->st_ctim.tv_nsec;
}

void convert_timeval_from_host(riscv::abi::timeval *guest_tv, struct timeval *host_tv) {
    guest_tv->tv_sec   = host_tv->tv_sec;
    guest_tv->tv_usec  = host_tv->tv_usec;
}

// When an error occurs during a system call, Linux will return the negated value of the error number. Library
// functions, on the other hand, usually return -1 and set errno instead.
// Helper for converting library functions which use global variable `errno` to carry error information to a linux
// syscall style which returns a negative value representing the errno.
emu::sreg_t return_errno(emu::sreg_t val) {
    if (val != -1) return val;
    return -static_cast<emu::sreg_t>(convert_errno_from_host(errno));
}

}

namespace emu {

reg_t syscall(
    State *state, riscv::abi::Syscall_number nr,
    reg_t arg0, reg_t arg1, reg_t arg2, [[maybe_unused]] reg_t arg3, [[maybe_unused]] reg_t arg4, [[maybe_unused]] reg_t arg5
) {
    bool strace = state->strace;

    switch (nr) {
        case riscv::abi::Syscall_number::close: {
            // Handle standard IO specially, pretending close is sucessful.
            sreg_t ret;
            if (arg0 == 1 || arg0 == 2) {
                ret = 0;
            } else {
                ret = return_errno(close(arg0));
            }

            if (strace) {
                util::log("close({}) = {}\n", arg0, ret);
            }

            return ret;
        }
        case riscv::abi::Syscall_number::write: {
            std::vector<std::byte> buffer(arg2);
            state->mmu->copy_to_host(arg1, buffer.data(), arg2);

            // Handle standard IO specially, since it is shared between emulator and guest program.
            sreg_t ret;
            if (arg0 == 1) {
                std::cout.write(reinterpret_cast<char*>(buffer.data()), arg2);
                std::cout << std::flush;
                ret = arg2;
            } else if (arg0 == 2) {
                std::cerr.write(reinterpret_cast<char*>(buffer.data()), arg2);
                std::cerr << std::flush;
                ret = arg2;
            } else {
                ret = return_errno(write(arg1, buffer.data(), arg2));
            }

            if (strace) {
                util::log("write({}, \"{}\", {}) = {}\n",
                    arg0,
                    escape((char*)buffer.data(), buffer.size()),
                    arg2,
                    ret
                );
            }

            return ret;
        }
        case riscv::abi::Syscall_number::fstat: {
            struct stat host_stat;
            sreg_t ret = return_errno(fstat(arg0, &host_stat));

            // When success, convert stat format to guest format.
            if (ret == 0) {
                struct riscv::abi::stat guest_stat;
                memset(&guest_stat, 0, sizeof(riscv::abi::stat));
                convert_stat_from_host(&guest_stat, &host_stat);
                state->mmu->copy_from_host(arg1, &guest_stat, sizeof(riscv::abi::stat));
            }

            if (strace) {
                if (ret == 0) {
                    util::log("fstat({}, {{st_mode={:#o}, st_size={}, ...}}) = 0\n", arg0, host_stat.st_mode, host_stat.st_size);
                } else {
                    util::log("fstat({}, {:#x}) = {}\n", arg0, arg1, ret);
                }
            }

            return ret;
        }
        case riscv::abi::Syscall_number::exit: {
            if (strace) {
                util::log("exit({}) = ?\n", arg0);
            }

            // Record the exit_code so that the emulator can correctly return it.
            throw emu::Exit_control { static_cast<uint8_t>(arg0) };
        }
        case riscv::abi::Syscall_number::gettimeofday: {
            struct timeval host_tv;

            // TODO: gettimeofday is obsolescent. Even if some applications require this syscall, we should try to work
            // around it instead of using the obsolescent function.
            sreg_t ret = return_errno(gettimeofday(&host_tv, nullptr));

            if (ret == 0) {
                struct riscv::abi::timeval guest_tv;
                memset(&guest_tv, 0, sizeof(riscv::abi::timeval));
                convert_timeval_from_host(&guest_tv, &host_tv);
                state->mmu->copy_from_host(arg0, &guest_tv, sizeof(riscv::abi::timeval));
            }

            if (strace) {
                if (ret == 0) {
                    util::log("gettimeofday({{{}, {}}}, NULL) = 0\n", host_tv.tv_sec, host_tv.tv_usec);
                } else {
                    util::log("gettimeofday({:#x}) = {}\n", arg0, ret);
                }
            }

            return ret;
        }
        case riscv::abi::Syscall_number::brk: {
            if (arg0 < state->original_brk) {
                // Cannot reduce beyond original_brk
            } else {
                reg_t new_heap_end = std::max(state->heap_start, (arg0 + page_mask) &~ page_mask);

                if (new_heap_end > state->heap_end) {

                    // The heap needs to be expanded
                    state->mmu->allocate_page(state->heap_end, new_heap_end - state->heap_end);
                    state->heap_end = new_heap_end;

                } else if (new_heap_end < state->heap_end) {

                    // TODO: Also shrink when brk is reduced.
                }

                state->brk = arg0;
            }
            reg_t ret = state->brk;
            if (strace) {
                util::log("brk({}) = {}\n", pointer(arg0), pointer(ret));
            }
            return ret;
        }
        default: {
            std::cerr << "illegal syscall " << static_cast<int>(nr) << std::endl;
            return -static_cast<sreg_t>(riscv::abi::Errno::enosys);
        }
    }
}

}

