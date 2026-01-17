//
//  Item.swift
//  polar-bear
//
//  Created by Yelarys Yertaiuly on 17/01/2026.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
