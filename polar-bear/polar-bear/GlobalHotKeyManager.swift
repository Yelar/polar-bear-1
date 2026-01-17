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
    private var isTriggered = false

    func start() {
        stop()
        monitor = NSEvent.addGlobalMonitorForEvents(matching: .flagsChanged) { [weak self] event in
            self?.handleFlagsChanged(event)
        }
    }

    func stop() {
        if let monitor {
            NSEvent.removeMonitor(monitor)
            self.monitor = nil
        }
        isTriggered = false
    }

    private func handleFlagsChanged(_ event: NSEvent) {
        let requiredFlags: NSEvent.ModifierFlags = [.command, .option]
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)

        if flags.contains(requiredFlags) {
            if !isTriggered {
                isTriggered = true
                DispatchQueue.main.async { [weak self] in
                    self?.onTrigger?()
                }
            }
        } else {
            isTriggered = false
        }
    }

    deinit {
        stop()
    }
}
