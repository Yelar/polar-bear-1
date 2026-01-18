//
//  GlobalHotKeyManager.swift
//  polar-bear
//
//  Created by Codex.
//

import AppKit

final class GlobalHotKeyManager {
    var onTrigger: (() -> Void)?

    private var monitor: Any?

    func start() {
        stop()
        monitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [weak self] event in
            self?.handleKeyDown(event)
        }
    }

    func stop() {
        if let monitor {
            NSEvent.removeMonitor(monitor)
            self.monitor = nil
        }
    }

    private func handleKeyDown(_ event: NSEvent) {
        if event.isARepeat { return }
        let requiredFlags: NSEvent.ModifierFlags = [.command, .option]
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        guard flags.contains(requiredFlags) else { return }
        guard event.charactersIgnoringModifiers?.lowercased() == "c" else { return }

        DispatchQueue.main.async { [weak self] in
            self?.onTrigger?()
        }
    }

    deinit {
        stop()
    }
}
