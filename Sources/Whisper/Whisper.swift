import Foundation
import whisper_cpp

public enum WhisperSamplingStrategy: UInt32 {
    case greedy = 0
    case beamSearch
}

@dynamicMemberLookup
public class WhisperParams {
    public static let `default` = WhisperParams(strategy: .greedy)

    internal var whisperParams: whisper_full_params
    private var stringPointers: [UnsafeMutablePointer<CChar>] = [] {
        willSet {
            // Find removed pointers and free them
            let removals = stringPointers.filter { !newValue.contains($0) }

            for pointer in removals {
                free(pointer)
            }
        }
    }

    public init(strategy: WhisperSamplingStrategy = .greedy) {
        self.whisperParams = whisper_full_default_params(whisper_sampling_strategy(rawValue: strategy.rawValue))
        self.language = .auto
        
        let maxThreads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        self.n_threads = Int32(maxThreads)
        
        self.max_len = 5
    }

    deinit {
        for pointer in stringPointers {
            free(pointer)
        }
    }

    public subscript<T>(dynamicMember keyPath: WritableKeyPath<whisper_full_params, T>) -> T {
        get { whisperParams[keyPath: keyPath] }
        set { whisperParams[keyPath: keyPath] = newValue }
    }

    public var language: WhisperLanguage {
        get { .init(rawValue: String(Substring(cString: whisperParams.language)))! }
        set {
            guard let pointer = strdup(newValue.rawValue) else { return }

            if let oldPointerIndex = stringPointers.firstIndex(where: { $0 == whisperParams.language }) {
                stringPointers.remove(at: oldPointerIndex) // stringPointers willSet takes care of freeing
            }

            stringPointers.append(pointer)
            whisperParams.language = UnsafePointer(pointer)
        }
    }
}

public struct Segment {
    public let startTime: Int
    public let endTime: Int
    public let text: String
}

public protocol WhisperDelegate {
    func whisper(_ aWhisper: Whisper, didUpdateProgress progress: Float)
    func whisper(_ aWhisper: Whisper, didProcessNewSegments segments: [Segment], atIndex index: Int)
    func whisper(_ aWhisper: Whisper, didCompleteWithSegments segments: [Segment])
    func whisper(_ aWhisper: Whisper, didErrorWith error: Error)
}

public extension WhisperDelegate {
    func whisper(_ aWhisper: Whisper, didUpdateProgress progress: Float) {
        //
    }

    func whisper(_ aWhisper: Whisper, didProcessNewSegments segments: [Segment], atIndex index: Int) {
        //
    }

    func whisper(_ aWhisper: Whisper, didCompleteWithSegments segments: [Segment]) {
        //
    }

    func whisper(_ aWhisper: Whisper, didErrorWith error: Error) {
        //
    }
}

public class Whisper {
    private let whisperContext: OpaquePointer

    public var delegate: WhisperDelegate?
    public var params: WhisperParams

    internal var isRunning = false
    private var retainedSelf: Unmanaged<Whisper>? {
        willSet {
            guard let retainedSelf else { return }

            retainedSelf.release() // Clear out old reference
        }
    }

    public init(fromFileURL fileURL: URL, withParams params: WhisperParams = .default) {
        self.whisperContext = fileURL.relativePath.withCString { whisper_init_from_file($0) }
        self.params = params

        self.setupCallbacks()
    }

    public init(fromData data: Data, withParams params: WhisperParams = .default) {
        var copy = data // TODO: Find way around copying memory

        self.whisperContext = copy.withUnsafeMutableBytes { whisper_init_from_buffer($0.baseAddress!, data.count) }
        self.params = params

        self.setupCallbacks()
    }

    deinit {
        whisper_free(whisperContext)
    }

    private func setupCallbacks() {
        let retainedSelf = Unmanaged.passRetained(self)
        self.retainedSelf = retainedSelf
        let opaqueSelf = retainedSelf.toOpaque()

        // All user data can share the same retained self for simplicity
        params.new_segment_callback_user_data = opaqueSelf
        params.progress_callback_user_data = opaqueSelf
        params.encoder_begin_callback_user_data = opaqueSelf

        params.new_segment_callback = { (ctx: OpaquePointer?, newSegmentCount: Int32, userData: UnsafeMutableRawPointer?) in
            guard let ctx, let userData else { return }
            let whisper = Unmanaged<Whisper>.fromOpaque(userData).takeUnretainedValue()
            guard let delegate = whisper.delegate else { return }

            let segmentCount = whisper_full_n_segments(ctx)
            var newSegments: [Segment] = []
            newSegments.reserveCapacity(Int(newSegmentCount))

            let startIndex = segmentCount - newSegmentCount

            for i in startIndex..<segmentCount {
                guard let text = whisper_full_get_segment_text(ctx, i) else { continue }
                let startTime = whisper_full_get_segment_t0(ctx, i)
                let endTime = whisper_full_get_segment_t1(ctx, i)

                newSegments.append(.init(
                    startTime: Int(startTime) * 10, // Time is given in ms/10, so correct for that
                    endTime: Int(endTime) * 10,
                    text: String(Substring(cString: text))
                ))
            }

            DispatchQueue.main.async {
                delegate.whisper(whisper, didProcessNewSegments: newSegments, atIndex: Int(startIndex))
            }
        }

        params.progress_callback = { (ctx: OpaquePointer?, progress: Float, userData: UnsafeMutableRawPointer?) in
            guard let userData else { return }
            let whisper = Unmanaged<Whisper>.fromOpaque(userData).takeUnretainedValue()
            guard let delegate = whisper.delegate else { return }

            DispatchQueue.main.async {
                delegate.whisper(whisper, didUpdateProgress: progress)
            }
        }

        params.encoder_begin_callback = { (ctx: OpaquePointer?, userData: UnsafeMutableRawPointer?) -> Bool in
            guard let userData else { return true }
            let whisper = Unmanaged<Whisper>.fromOpaque(userData).takeUnretainedValue()

            return whisper.isRunning
        }
    }

    public func transcribe(audioFrames: [Float], completionHandler: @escaping (Result<[Segment], Error>) -> Void) {
        isRunning = true

        DispatchQueue.global(qos: .userInitiated).async { [unowned self] in

            whisper_full(whisperContext, params.whisperParams, audioFrames, Int32(audioFrames.count))

            let segmentCount = whisper_full_n_segments(whisperContext)

            var segments: [Segment] = []
            segments.reserveCapacity(Int(segmentCount))

            for i in 0..<segmentCount {
                guard let text = whisper_full_get_segment_text(whisperContext, i) else { continue }
                let startTime = whisper_full_get_segment_t0(whisperContext, i)
                let endTime = whisper_full_get_segment_t1(whisperContext, i)

                segments.append(
                    .init(
                        startTime: Int(startTime) * 10, // Correct for ms/10
                        endTime: Int(endTime) * 10,
                        text: String(Substring(cString: text))
                    )
                )
            }

            DispatchQueue.main.async {
                self.delegate?.whisper(self, didCompleteWithSegments: segments)
                completionHandler(.success(segments))
            }
        }
    }

    public func stop() {
        isRunning = false
    }

    @available(iOS 13, macOS 10.15, *)
    public func transcribe(audioFrames: [Float]) async throws -> [Segment] {
        return try await withCheckedThrowingContinuation { cont in
            self.transcribe(audioFrames: audioFrames) { result in
                switch result {
                case .success(let segments):
                    cont.resume(returning: segments)
                case .failure(let error):
                    cont.resume(throwing: error)
                }
            }
        }
    }
}
