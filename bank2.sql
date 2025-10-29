/*
 Navicat Premium Data Transfer

 Source Server         : 阿里云 - 企业信贷客户库
 Source Server Type    : MySQL
 Source Server Version : 80025 (8.0.25)
 Source Host           : rm-uf6z891lon6dxuqblqo.mysql.rds.aliyuncs.com:3306
 Source Schema         : bank2

 Target Server Type    : MySQL
 Target Server Version : 80025 (8.0.25)
 File Encoding         : 65001

 Date: 29/10/2025 23:02:33
*/

SET NAMES utf8mb4;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for customer_data
-- ----------------------------
DROP TABLE IF EXISTS `customer_data`;
CREATE TABLE `customer_data`  (
  `customer_id` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NOT NULL COMMENT '客户编号',
  `gender` char(1) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '性别: M-男, F-女',
  `age` int NULL DEFAULT NULL COMMENT '年龄',
  `occupation` varchar(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '职业',
  `marital_status` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '婚姻状况: 已婚、未婚、离异',
  `city_level` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '城市等级: 一线、二线、三线',
  `account_open_date` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '开户日期',
  `total_aum` decimal(18, 2) NULL DEFAULT NULL COMMENT '总资产管理规模',
  `deposit_balance` decimal(18, 2) NULL DEFAULT NULL COMMENT '存款余额',
  `wealth_management_balance` decimal(18, 2) NULL DEFAULT NULL COMMENT '理财余额',
  `fund_balance` decimal(18, 2) NULL DEFAULT NULL COMMENT '基金余额',
  `insurance_balance` decimal(18, 2) NULL DEFAULT NULL COMMENT '保险余额',
  `deposit_balance_monthly_avg` decimal(18, 2) NULL DEFAULT NULL COMMENT '存款月均余额',
  `wealth_management_balance_monthly_avg` decimal(18, 2) NULL DEFAULT NULL COMMENT '理财月均余额',
  `fund_balance_monthly_avg` decimal(18, 2) NULL DEFAULT NULL COMMENT '基金月均余额',
  `insurance_balance_monthly_avg` decimal(18, 2) NULL DEFAULT NULL COMMENT '保险月均余额',
  `monthly_transaction_count` decimal(10, 2) NULL DEFAULT NULL COMMENT '月均交易次数',
  `monthly_transaction_amount` decimal(18, 2) NULL DEFAULT NULL COMMENT '月均交易金额',
  `last_transaction_date` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '最近交易日期',
  `mobile_bank_login_count` int NULL DEFAULT NULL COMMENT '手机银行登录次数',
  `branch_visit_count` int NULL DEFAULT NULL COMMENT '网点访问次数',
  `last_mobile_login` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '最近手机银行登录日期',
  `last_branch_visit` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '最近网点访问日期',
  `customer_tier` varchar(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci NULL DEFAULT NULL COMMENT '客户等级: 普通、潜力、临界、高净值',
  PRIMARY KEY (`customer_id`) USING BTREE,
  INDEX `idx_customer_tier`(`customer_tier` ASC) USING BTREE,
  INDEX `idx_age`(`age` ASC) USING BTREE,
  INDEX `idx_total_aum`(`total_aum` ASC) USING BTREE,
  INDEX `idx_occupation`(`occupation` ASC) USING BTREE,
  INDEX `idx_city_level`(`city_level` ASC) USING BTREE,
  INDEX `idx_account_open_date`(`account_open_date` ASC) USING BTREE,
  INDEX `idx_last_transaction_date`(`last_transaction_date` ASC) USING BTREE
) ENGINE = InnoDB CHARACTER SET = utf8mb4 COLLATE = utf8mb4_0900_ai_ci COMMENT = '银行客户数据表' ROW_FORMAT = Dynamic;

SET FOREIGN_KEY_CHECKS = 1;
